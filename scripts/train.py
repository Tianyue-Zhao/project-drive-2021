import numpy as np
import rospy
import parser
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

#collision detection function
#returns true if it determines the car has crashed
#this will be used to trigger a simulator reset
def col_detect(state):

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

#"Main function" here
#Handle flags from command line
#Call "train()" with --train flag
#Call "run()" with --run flag
#Prepare for later additions such as "--steps=1000" and so on
#Process these flags and save the inputs to variables