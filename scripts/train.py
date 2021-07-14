import numpy as np
import rospy
import parser
import json
import time
import argparse
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool
from tf_environment import PD_Environment
from gym_environment import Gym_Environment
from network import custom_network
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce import Runner
from f1tenth_gym_ros.msg import RaceInfo


#One state object is declared in train.py
#This is passed by reference to all listeners and the tf environment
#Includes all RL state information and auxiliary info
#Updated by listeners
class State:
    def __init__(self):
        #initialize state with empty variables
        self.line_scan = np.zeros(1080)
        #x, y, theta, velocity, angular_vel
        #are odometry variables
        #x, y is car's position on the map
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.velocity = 0.0
        self.angular_vel = 0.0
        #True for crash
        self.crash_det = False
        self.lap_finish = False
        self.lap_time = 0.0
        self.configs = 0.0
        self.col_counter = 0
        self.v_counter = 0
        #The lap count when last checked
        self.prev_lap = 0
        #Lap counting with waypoints
        self.prev_waypoint = 0
        self.cur_waypoint = 0
        self.num_waypoints = 0
        self.turn_back = False
        self.ep_reward = 0.0
        self.default_reward = 0.0
        self.ds_reward = False #Reward for approaching the next waypoint
        #Apr 28 addition of continuous rewards
        self.prev_distance = 0.0
        self.cur_distance = 0.0
        #Give configuration to tf_environment
        self.verbose = False
        #Publisher for distribution display
        self.st_display_pub = 0.0
        self.entropy_reg = 0.0
        #Action taken for the action distribution
        self.cur_steer = 0

#collision detection function
#returns true if it determines the car has crashed
#this will be used to trigger a simulator reset
#This is used for testing and evaluation
#along with the test() function
def col_detect(main_state):
    dist_threshold = main_state.configs["DIST_THR"]    #this is the threshold to determine if an object is
                            #so close as to hit the agent
    v_threshold = main_state.configs["VEL_THR"]      #this is the threshold to determine if the agent is
                            #"stopped" or not
    found = np.any(np.less(main_state.line_scan, dist_threshold))
    #if none of the points are too close, no crash detected
    if found:
        main_state.col_counter += 1
    else:
        #if no crash detected, reset the counter
        main_state.col_counter = 0

    #if the agent is "stopped" increment counter, else reset it
    if main_state.velocity < v_threshold:
        main_state.v_counter += 1
    else:
        main_state.v_counter = 0

    #This variable determines how many crash values must be found before a crash is declared
    main_state.crash_det = ((main_state.col_counter > main_state.configs['CRASH_THR']) or (main_state.v_counter > main_state.configs['CRASH_THR']))

    return main_state.crash_det

def test(flags):
    rospy.init_node("rl_algorithm", anonymous = True)
    main_state = State()
    config_file = open('configs/config.json')
    main_state.configs = json.load(config_file)
    config_file.close()
    #Publishers
    drive_announce = rospy.Publisher(main_state.configs['CONTROL_TOPIC'], AckermannDriveStamped, queue_size=1)
    reset_announce = rospy.Publisher(main_state.configs['RESET_TOPIC'], Bool, queue_size=1)
    main_state.st_display_pub = rospy.Publisher(main_state.configs['ST_DISPLAY_TOPIC'], Marker, queue_size=10)

    #Flags for testing
    if(flags.steps):
        test_steps = flags.steps
    else:
        print("The number of steps must be specified")
        return
    if(flags.verbose):
        main_state.verbose = True

    environment = PD_Environment(reset_announce, drive_announce, main_state)
    environment.publish_markers()

    #Initialize agent
    #TODO: Consolidate into configs
    agent = Agent.create(agent = "ppo", network = custom_network(),
        batch_size = 5, environment = environment,
        max_episode_timesteps = 2000, tracking = "all")
    if(flags.load):
        files = flags.laod.split('/')
        if(len(files) > 1):
            agent = Agent.load(directory = files[0], filename = files[1],
                environment = environment, agent = agent)
        else:
            agent = Agent.load(directory = flags.load, environment = environment,
                agent = agent)
    else:
        print("A load file must be specified")
        return
    
    #Define the tracking tensor names
    ST_TENSOR = 'agent/policy/turning_angle_distribution/probabilities'
    for i in range(test_steps):
        num_steps = 0
        environment.reset()
        states = parser.assemble_state(main_state)
        done = False
        main_state.crash_deet = False
        main_state.lap_finish = False

        while not done and num_steps < 2000:
            num_steps += 1
            actions = agent.act(states)
            all_probs = agent.tracked_tensors()
            parser.publish_steering_prob(all_probs[ST_TENSOR],
                main_state.st_display_pub, main_state.cur_steer)
            states, done, reward = environment.execute(actions = actions)
            col_detect(main_state)
            if(num_steps < 10):
                done = False
            if(main_state.crash_det):
                print("Crashed")
            if(main_state.lap_finish):
                print("Lap finished")
        print("Episode {} done after {}".format(i, num_steps))

#Parameters such as save path, steps to train
#and load from path to be added later
#Now train() is used instead. The GUI is used for testing visualization
def train_GUI(flags):
    #Initialize node
    rospy.init_node("rl_algorithm", anonymous=True)
    #Initialize subscribers for laser and odom
    main_state = State()
    #TODO: Put config file into flags

    #Load config
    config_file = open('configs/config.json')
    main_state.configs = json.load(config_file)
    config_file.close()
    #Subscribers and Publishers
    laser_listen = rospy.Subscriber(main_state.configs['LASER_TOPIC'], LaserScan, parser.laser_parser, main_state, queue_size=1)
    odom_listen = rospy.Subscriber(main_state.configs['ODOM_TOPIC'], Odometry, parser.odom_parser, main_state, queue_size=1)
    info_listen = rospy.Subscriber(main_state.configs['INFO_TOPIC'], RaceInfo, parser.info_parser, main_state, queue_size=1)
    drive_announce = rospy.Publisher(main_state.configs['CONTROL_TOPIC'], AckermannDriveStamped, queue_size=1)
    reset_announce = rospy.Publisher(main_state.configs['RESET_TOPIC'], Bool, queue_size=1)
    main_state.st_display_pub = rospy.Publisher(main_state.configs['ST_DISPLAY_TOPIC'], Marker, queue_size=10)
    #Publish True to reset_announce to reset the simulator

    #Accept flag params
    if(flags.steps):
        train_steps = flags.steps
    else:
        train_steps = main_state.configs["NUM_RUNS_TOT"]
    if(flags.save):
        save_file = flags.save
    else:
        save_file = main_state.configs["MODEL_DIR"]
    if(flags.verbose):
        main_state.verbose = True
    if(not flags.lap_time):
        main_state.default_reward = 0.01
    if(flags.entropy):
        main_state.entropy_reg = flags.entropy
    else:
        main_state.entropy_reg = main_state.configs["DEF_ENTROPY"]
    if(flags.ds_reward):
        main_state.ds_reward = True
    else:
        main_state.ds_reward = False

    # Initialize environment
    # TODO: Define max_episode_timesteps from CONFIG file
    #environment = Environment.create(
    #    environment=PD_Environment, max_episode_timesteps=100
    #)
    environment = PD_Environment(reset_announce, drive_announce, main_state)
    environment.publish_markers()

    # Initialize Agent
    agent = Agent.create(agent="ppo", network=custom_network(),
        batch_size = 5, entropy_regularization = main_state.entropy_reg,
        environment=environment, max_episode_timesteps=2000,
        learning_rate = 0.002,
        #tracking="all")
        tracking="all", summarizer=main_state.configs["SUM_DIR"])
    if(flags.load):
        files = flags.load.split('/')
        if(len(files)>1):
            agent = Agent.load(directory=files[0], filename=files[1], environment=environment,
                agent=agent)
        else:
            agent = Agent.load(directory=flags.load,environment=environment, agent=agent)
        print("Agent loaded from "+flags.load)
    #The agent network configuration could be printed with agent.get_architecture()

    # Run the save loop
    for i in range(int((train_steps-1)/main_state.configs["SAVE_RUNS"])+1):
        run(environment, agent, main_state, main_state.configs["SAVE_RUNS"], 10000, False)
        agent.save(save_file, format="checkpoint", append="episodes")

def train(flags):
    main_state = State()
    #Load config
    config_file = open('configs/config.json')
    main_state.configs = json.load(config_file)
    config_file.close()
    #Accept flag params
    if(flags.steps):
        train_steps = flags.steps
    else:
        train_steps = main_state.configs["NUM_RUNS_TOT"]
    if(flags.save):
        save_file = flags.save
    else:
        save_file = main_state.configs["MODEL_DIR"]
    if(flags.verbose):
        main_state.verbose = True
    if(not flags.lap_time):
        main_state.default_reward = 0.01
    if(flags.entropy):
        main_state.entropy_reg = flags.entropy
    else:
        main_state.entropy_reg = main_state.configs["DEF_ENTROPY"]
    if(flags.ds_reward):
        main_state.ds_reward = True
    else:
        main_state.ds_reward = False
    environments = list()
    for i in range(8):
        environments.append(Gym_Environment(main_state))
    #Initialize the agent
    agent = Agent.create(agent = "ppo", network = custom_network(),\
        environment = environments[0], max_episode_timesteps=2000,\
        parallel_interactions = 8,\
        learning_rate = 0.002, summarizer = main_state.configs["SUM_DIR"],\
        batch_size = 10, entropy_regularization = main_state.entropy_reg)
    if(flags.load):
        files = flags.load.split('/')
        if(len(files) > 1):
            agent = Agent.load(directory = files[0], filename = files[1],\
                max_episode_timesteps = 2000, learning_rate = 0.002,\
                summarizer = main_state.configs["SUM_DIR"],\
                batch_size = 10,\
                environment = environments[0], agent = agent)
        else:
            agent = Agent.load(directory = files[0], environment = environments[0],\
                max_episode_timesteps = 2000, learning_rate = 0.002,\
                summarizer = main_state.configs["SUM_DIR"],\
                batch_size = 10,\
                agent = agent)
    runner = Runner(agent = agent, environments = environments, num_parallel = 8,\
        remote = 'multiprocessing')
    if(train_steps <= main_state.configs["SAVE_RUNS"]):
        runner.run(num_episodes = train_steps, batch_agent_calls = True)
        agent.save(save_file, format = "checkpoint", append = "episodes")
    else:
        for i in range(int((train_steps - 1) / main_state.configs["SAVE_RUNS"]) + 1):
            runner.run(num_episodes = main_state.configs["SAVE_RUNS"], batch_agent_calls = True)
            agent.save(save_file, format = "checkpoint", append = "episodes")

#Train for n episodes
def run(environment, agent, main_state, num_episodes, max_step_per_epi, test=False):
    #Define the tracking tensor names
    ST_TENSOR = 'agent/policy/turning_angle_distribution/probabilities'
    #Run through each episode
    for i in range(num_episodes):
        num_steps = 0
        environment.reset()
        states = parser.assemble_state(main_state)
        done = False
        main_state.crash_det = False
        main_state.lap_finish = False
        
        #Run through the episode
        while not done and num_steps < max_step_per_epi:
            num_steps +=1
            if(main_state.verbose):
                start_time = time.time()
            actions = agent.act(states=states)
            all_probs = agent.tracked_tensors()
            parser.publish_steering_prob(all_probs[ST_TENSOR],
                main_state.st_display_pub, main_state.cur_steer)
            states, done, reward = environment.execute(actions=actions)
            col_detect(main_state)
            if(num_steps<10):
                done = False
            if(main_state.crash_det):
                print("Crashed")
            if(main_state.lap_finish):
                print("Lap finished")
                main_state.lap_finish = False
            #Only update model if not testing
            if not test:
                agent.observe(terminal=done, reward=reward)
            if(main_state.verbose):
                print("Step decision: "+str(time.time() - start_time))

        print("Episode {} done after {}".format(i,num_steps))

if __name__ == "__main__":
    #Handle flags from command line
    arg_parser = argparse.ArgumentParser()

    #Set flags
    #Prepare for later additions such as "--steps=1000" and so on
    arg_parser.add_argument("--train", help="Begin training model", action="store_true")
    arg_parser.add_argument("--test", help="GUI Testing", action="store_true")
    arg_parser.add_argument("--run", help="Run program", action="store_true")
    arg_parser.add_argument("--steps", type=int, help="Add number of steps to train model")
    arg_parser.add_argument("--save", type=str, help="Add save file path")
    arg_parser.add_argument("--load", type=str, help="Add load file path")
    arg_parser.add_argument("--verbose", help="Slow mode to diagnose decisions", action="store_true")
    #Option to focus on lap time and stop giving rewards for each time step
    arg_parser.add_argument("--lap_time", help="Mode to focus on lap time", action="store_true")
    #Option to adjust the entropy regularization term. Higher value encourages exploration.
    arg_parser.add_argument("--entropy", type=float, help="Entropy regularization term")
    #Option to enable rewards for approaching the next waypoint, based on distance
    arg_parser.add_argument("--ds_reward", help="Reward for approaching the next waypoint", action="store_true")

    #Process these flags
    flags = arg_parser.parse_args()

    #Train model
    if flags.train:
        train(flags)
    elif flags.test:
        test(flags)
    
