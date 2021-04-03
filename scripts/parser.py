from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
import numpy as np
import math


def laser_parser(scan_data, state):
    #Assembles all points detected by LiDAR in an array
    #Each point described by x,y coordinates
    #These coordinates are from the perspective of the car
    #These two callbacks are called by ROS to update the state
    laser_points = []
    ranges = scan_data.ranges
    angle_min = scan_data.angle_min
    angle_max = scan_data.angle_max
    angle_increment = scan_data.angle_increment
    for i in range(len(ranges)):
        x = ranges[i] * math.sin(angle_min + i * angle_increment)
        y = ranges[i] * math.cos(angle_min + i * angle_increment)
        laser_points.append([x, y])
    state.cur_points = np.asarray(laser_points)

def odom_parser(data, state):
    #data is the odom message from ROS
    #state is the state object to save to
    state.x = data.pose.pose.position.x
    state.y = data.pose.pose.position.y
    quaternion = data.pose.pose.orientation
    quaternion = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    #translate quaternion to euler coordinates
    #and then save the angle about the z axis to theta
    state.theta = euler_from_quaternion(quaternion)[2]
    state.velocity = data.twist.twist.linear.x
    state.angular_vel = data.twist.twist.angular.z

def info_parser(data, state):
    state.lap_time = data.ego_elapsed_time

def assemble_state(main_state):
    cur_state = np.asarray([
        main_state.x,
        main_state.y,
        main_state.theta,
        main_state.velocity,
        main_state.angular_vel
    ])
    cur_state = np.concatenate((cur_state, np.reshape(main_state.cur_points, 2160)))
    return cur_state