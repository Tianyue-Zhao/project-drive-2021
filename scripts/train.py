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
        self.cur_points = np.zeros((2,1))
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

#collision detection function
#returns true if it determines the car has crashed
#this will be used to trigger a simulator reset
def col_detect(state):
    dist_threshold = 0.2    #this is the threshold to determine if an object is
                            #so close as to hit the agent
    v_threshold = 0.01      #this is the threshold to determine if the agent is
                            #"stopped" or not
    found = False
    #this loop iterates over each point in the LaserScan and finds the distance
    #if the distance is smaller than the threshold it updates the found variable
    #TODO: can be streamlined with Numpy. Ok for now.
    for point in state.cur_points:
        x = point[0]
        y = point[1]
        dist = math.sqrt(x**2 + y**2)
        if dist < dist_threshold :
            found = True
    #if none of the points are too close, no crash detected
    if found:
        state.col_counter += 1
    else:
        #if no crash detected, reset the counter
        state.col_counter = 0

    #if the agent is "stopped" increment counter, else reset it
    if state.velocity < v_threshold:
        state.v_counter += 1
    else:
        state.v_counter = 0

    #This variable determines how many crash values must be found before a crash is declared
    state.crash_det = ((state.col_counter > state.configs['crash_thr']) or (state.v_counter > state.configs['CRASH_THR']))

    return state.crash_det

        

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
    #TODO: implement load functionality

    # Initialize environment
    # TODO: Define max_episode_timesteps from CONFIG file
    #environment = Environment.create(
    #    environment=PD_Environment, max_episode_timesteps=100
    #)
    environment = PD_Environment(reset_announce, drive_announce, main_state)

    # Initialize Agent
    agent = Agent.create(agent='configs/agent_config.json', environment=environment)

    # Run the save loop
    for i in range(int((train_steps-1)/main_state.configs["SAVE_RUNS"])+1):
        run(environment, agent, main_state, main_state.configs["SAVE_RUNS"], 1000, False)
        agent.save(save_file, format="hdf5", append="episodes")

def assemble_state(main_state):
    cur_state = np.asarray([
        main_state.x,
        main_state.y,
        main_state.theta,
        main_state.velocity,
        main_state.angular_vel
    ])
    return cur_state

#Train for n episodes
def run(environment, agent, main_state, num_episodes, max_step_per_epi, test=False):
    #Run through each episode
    for i in range(num_episodes):
        num_steps = 0
        environment.reset()
        states = assemble_state(main_state)
        print(states)
        internals = agent.initial_internals()
        done = False
        
        #Run through the episode
        while not done and num_steps < max_step_per_epi:
            num_steps +=1
            actions = agent.act(states=states)
            states, done, reward = environment.execute(actions=actions)
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

    #Process these flags
    flags = arg_parser.parse_args()

    #Train model
    if flags.train:
        train(flags)
    
