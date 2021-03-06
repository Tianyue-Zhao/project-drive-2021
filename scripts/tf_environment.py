import random
import rospy
import numpy as np
import time
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from f1tenth_gym_ros.msg import RaceInfo
from tensorforce import Environment

# TODO: consider moving these parameters to train.py
# time delay to get next step. In seconds
NEXT_STATE_DELAY = 0.5
# dummpy control topic
CONTROL_TOPIC = "/drive"
# dummy odom topic
ODOM_TOPIC = "/odom"
# dummy frame ID
DRIVE_FRAME = "drive"

class PD_Environment(Environment):
    # The custom Tensorforce environment used for training

    # RS_pub is the publisher to the reset channel
    # D_pub is the publisher to the drive channel
    # Both are already initialized in train.py
    def __init__(self, RS_pub, D_pub, main_state):
        self.RS = RS_pub
        self.D = D_pub
        # next state of the agent after taking an action
        self.main_state = main_state
        # ros rate that controls the frequency of reading messages.
        self.rate = rospy.Rate(1.0 / NEXT_STATE_DELAY)
        # Actions to be taken by agent, containing two categories of actions:
        # velocity and turning angles.
        # TODO: consider other formats for actions
        self.agent_actions = self.action_values()
        # publisher for velocity and turning angles
        self.ack_pub = rospy.Publisher(
            CONTROL_TOPIC, AckermannDriveStamped, queue_size=1
        )

    def action_values(self):
        """ Helper function that creates a dictionary of actions for the agent 
        to choose from.

        :return: dictionary of actions.
        :rtype: dict of str : list (float)
        """
        agent_actions = {"velocity": [], "turning_angle": []}
        # velocity and turn_ang are currently lists for convenience.
        agent_actions["velocity"] = np.arange(
            self.main_state.configs['VLOW'], self.main_state.configs['VHIGH'], (self.main_state.configs['VHIGH'] - self.main_state.configs['VLOW']) / self.main_state.configs['NUM_VEL_CHOICES'] 
        ).tolist()
        agent_actions["turning_angle"] = np.arange(
            self.main_state.configs['ANGL'],
            self.main_state.configs['ANGR'],
            (self.main_state.configs['ANGR'] - self.main_state.configs['ANGL']) / self.main_state.configs['NUM_TURN_ANG'],
        ).tolist()
        print(agent_actions)
        return agent_actions

    def actions(self):
        agent_actions = {"velocity":
            {"type": "int", "num_values": self.main_state.configs["NUM_VEL_CHOICES"]},
            "turning_angle": {"type": "int", "num_values": self.main_state.configs["NUM_TURN_ANG"]}}
        return agent_actions

    # A terminal state reached if the car has crashed
    # or a lap had been finished
    def terminal(self):
        return self.main_state.crash_det or self.main_state.lap_finish

    def get_next_state(self, actions):
        """ Helper function for execute. publishes velocity and turning angle 
        to the f1tenth gym environment, and then uses the condition of the car a 
        constant time after vel and turn_ang have beenapplied as the next state.
        It currently randomly chooses the action.
        The constant time delay currently is 0.5 sec. Yet to be tested
        
        :param actions: A dictionary of actions to be chosen from
        :type actions: dict of str : list (float)

        """
        # TODO: Need to test if 0.5 sec delay works
        # TODO: FIgure out better way to choose actions.
        #vel = actions["velocity"][random.randint(0, NUM_VEL_CHOICES - 1)]
        #steer_ang = actions["turning_angle"][random.randint(0, NUM_TURN_ANG - 1)]
        vel = self.agent_actions["velocity"][actions["velocity"]]
        steer_ang = self.agent_actions["turning_angle"][actions["turning_angle"]]
        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = rospy.Time.now()
        ack_msg.header.frame_id = DRIVE_FRAME
        ack_msg.drive.steering_angle = steer_ang
        ack_msg.drive.speed = vel
        self.ack_pub.publish(ack_msg)
        # use ros rate to wait for 0.5 sec before reading the next odometry
        # reading is automatically handled by odom_callback.
        # TODO: test whether the ros rate solution works
        # NOTE: rate.sleep still pauses the entire thread. May affect other funcs
        self.rate.sleep()

    def execute(self, actions):
        """Overriden execute function that takes in an action and "advances the 
        environment by one step." 

        :param actions: Dictionary of actions 
        :type actions: dict of str : list (float)
        :return: next state of the agent, whether the environment should be
        terminated, reward value
        :rtype: obj:State, bool or 2, float
        """
        print(actions)
        self.get_next_state(actions)
        cur_state = np.asarray([
            self.main_state.x,
            self.main_state.y,
            self.main_state.theta,
            self.main_state.velocity,
            self.main_state.angular_vel
        ])
        reward = self.reward()
        # currently using the given terminal method.
        # TODO:handle return option 2 (environment aborted)
        terminal = self.terminal()
        return cur_state, terminal, reward

    def states(self):
        return {'type': 'float', 'shape': (5,)}

    # This should override whatever default close function these is
    # Publish a message for the simulator to reset, and wait
    def close(self):
        message = Bool()
        message.data = True
        self.RS.publish(message)
        time.sleep(self.main_state.configs["RS_WAIT"])
        super().close()

    def reset(self):
        message = Bool()
        message.data = True
        self.RS.publish(message)
        time.sleep(self.main_state.configs["RS_WAIT"])

    #Basic reward function
    #Small punishment for crashing
    def reward(self):
        lap_finished = self.main_state.lap_finish
        lap_time = self.main_state.lap_time

        if lap_finished:
            reward = np.exp(-lap_time/self.main_state.configs["RW_MLT"])
        elif(self.main_state.crash_det):
            reward = -1
        else:
            reward = 0.01

        return reward
