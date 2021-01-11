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

#One state object is declared in train.py
#This is passed by reference to all listeners and the tf environment
#Includes all RL state information and auxiliary info
#Updated by listeners
class State:
    def __init__(self):
        #initialize state with empty variables
        self.cur_points = np.zeros((2,1))
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.velocity = 0.0
        self.angular_vel = 0.0
        self.crash_det = False
        self.lap_finish = False
        self.lap_time = 0.0
        self.configs = 0.0
        self.col_counter = 0
        self.v_counter = 0

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
    state.crash_det = ((state.col_counter > state.configs['crash_thr']) or (state.v_counter > state.configs['crash_thr']))

    return state.crash_det



#Parameters such as save path, steps to train
#and load from path to be added later
#Primary train function
def train():
    #Initialize node
    rospy.init_node("rl_algorithm", anonymous=True)
    #Initialize subscribers for laser and odom
    main_state = State()
    laser_listen = rospy.Subscriber(LASER_TOPIC, LaserScan, parser.laser_parser, main_state, queue_size=1)
    odom_listen = rospy.Subscriber(ODOM_TOPIC, Odometry, parser.odom_parser, main_state, queue_size=1)
    drive_announce = rospy.Publisher(CONTROL_TOPIC, AckermannDriveStamped, queue_size=1)
    reset_announce = rospy.Publisher(RESET_TOPIC, Bool, queue_size=1)
    #Publish True to reset_announce to reset the simulator
    #Load config
    config_file = open(CONFIG_FILE)
    main_state.configs = json.load(config_file)
    config_file.close()

if __name__ == "__main__":
    #Handle flags from command line
    parser = argparse.ArgumentParser()

    #Set flags
    #Prepare for later additions such as "--steps=1000" and so on
    parser.add_argument("--train", type=bool, help="Begin training model", action="store_true")
    parser.add_argument("--run", type=bool, help="Run program", action="store_true")
    parser.add_argument("--steps", type=int, help="Add number of steps to train model", action="store_true")
    parser.add_argument("--save", type=str, help="Add save file path", action="store_true")
    parser.add_argument("--load", type=str, help="Add load file path", action="store_true")

    #Process these flags
    args = parser.parse_args()

    #Save to variables
    if args.steps:
        steps = args.steps
    if args.save:
        save_file = args.save
    if args.load:
        load_file = args.load

    #Train model
    if args.train:
        train()
    
