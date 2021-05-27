import numpy as np
import math
import rospy
import parser
import json
import argparse
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from tf_environment import PD_Environment
from network import custom_network
from tensorforce.environments import Environment
from tensorforce.agents import Agent
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
        #Apr 28 addition of continuous rewards
        self.prev_distance = 0.0
        self.cur_distance = 0.0
        #Give configuration to tf_environment
        self.verbose = False

#collision detection function
#returns true if it determines the car has crashed
#this will be used to trigger a simulator reset
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

        

#Parameters such as save path, steps to train
#and load from path to be added later
#Primary train function
def train(flags):
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
    #TODO: implement load functionality

    # Initialize environment
    # TODO: Define max_episode_timesteps from CONFIG file
    #environment = Environment.create(
    #    environment=PD_Environment, max_episode_timesteps=100
    #)
    environment = PD_Environment(reset_announce, drive_announce, main_state)
    environment.publish_markers()

    # Initialize Agent
    agent = Agent.create(agent="ppo", network=custom_network(),
        batch_size = 5,
        environment=environment, max_episode_timesteps=2000)
    if(flags.load):
        agent = Agent.load(directory=flags.load,environment=environment, agent=agent)
        print("Agent loaded from "+flags.load)
    #Steps to output the default network configuration
    #print(agent.model.policy.network.layers)
    #print(agent.model.policy.network.layers[0].input_spec)
    #print(agent.model.policy.network.layers[0].size)
    #print(agent.model.policy.network.layers[1].input_spec)
    #print(agent.model.policy.network.layers[1].size)
    #print(agent.model.policy.distributions)

    # Run the save loop
    for i in range(int((train_steps-1)/main_state.configs["SAVE_RUNS"])+1):
        run(environment, agent, main_state, main_state.configs["SAVE_RUNS"], 10000, False)
        agent.save(save_file, format="checkpoint", append="episodes")

#Train for n episodes
def run(environment, agent, main_state, num_episodes, max_step_per_epi, test=False):
    #Run through each episode
    for i in range(num_episodes):
        num_steps = 0
        environment.reset()
        states = parser.assemble_state(main_state)
        internals = agent.initial_internals()
        done = False
        main_state.crash_det = False
        main_state.lap_finish = False
        
        #Run through the episode
        while not done and num_steps < max_step_per_epi:
            num_steps +=1
            actions = agent.act(states=states)
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

        print("Episode {} done after {}".format(i,num_steps))

if __name__ == "__main__":
    #Handle flags from command line
    arg_parser = argparse.ArgumentParser()

    #Set flags
    #Prepare for later additions such as "--steps=1000" and so on
    arg_parser.add_argument("--train", help="Begin training model", action="store_true")
    arg_parser.add_argument("--run", help="Run program", action="store_true")
    arg_parser.add_argument("--steps", type=int, help="Add number of steps to train model")
    arg_parser.add_argument("--save", type=str, help="Add save file path")
    arg_parser.add_argument("--load", type=str, help="Add load file path")
    arg_parser.add_argument("--verbose", help="Slow mode to diagnose decisions", action="store_true")

    #Process these flags
    flags = arg_parser.parse_args()

    #Train model
    if flags.train:
        train(flags)
    
