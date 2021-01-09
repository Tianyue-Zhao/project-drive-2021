import numpy as np
import math
import rospy
import parser
import json
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from tf_environment import PD_Environment

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

#collision detection function
#returns true if it determines the car has crashed
#this will be used to trigger a simulator reset
def col_detect(state):
    dist_threshold = 0.2    #this is the threshold to determine if an object is
                            #so close as to hit the agent
    #this loop iterates over each point in the LaserScan and finds the distance
    #if the distance is smaller than the threshold it announces a crash
    for point in state.cur_points:
        x = point[0]
        y = point[1]
        dist = math.sqrt(x**2 + y**2)
        if dist < dist_threshold :
            state.crash_det = True
            return True
    #if none of the points are too close, no crash detected
    return False



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

#"Main function" here
#Handle flags from command line
#Call "train()" with --train flag
#Call "run()" with --run flag
#Prepare for later additions such as "--steps=1000" and so on
#Process these flags and save the inputs to variables
