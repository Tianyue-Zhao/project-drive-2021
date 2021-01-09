import rospy
import time
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

class PD_Environment:
    #The custom Tensorforce environment used for training

    #RS_pub is the publisher to the reset channel
    #D_pub is the publisher to the drive channel
    #Both are already initialized in train.py
    def __init__(self, RS_pub, D_pub, main_state):
        self.RS = RS_pub
        self.D = D_pub
        self.main_state = main_state

    def states(self):
        return {'typee': 'float', 'shape': (5,)}

    #A terminal state reached if the car has crashed
    #or a lap had been finished
    def terminal(self):
        return self.main_state.crash_det or self.main_state.lap_finish

    #This should override whatever default close function there is
    #Publish a message for the simulator to reset, and wait
    def close(self):
        message = Bool()
        message.data = True
        self.RS.publish(message)
        time.sleep(self.main_state.configs["RS_wait"])