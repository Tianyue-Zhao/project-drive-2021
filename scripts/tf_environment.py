import rospy
from ackermann_msgs.msg import AckermannDriveStamped

class PD_Environment:
    #The custom Tensorforce environment used for training

    #RS_pub is the publisher to the reset channel
    #D_pub is the publisher to the drive channel
    def __init__(self, RS_pub, D_pub):
        self.RS = RS_pub
        self.D = D_pub