from sensor_msgs.msg import LaserScan
import numpy as np
import math


def laser_parser(scan_data):
    laser_points = []
    ranges = scan_data.ranges
    angle_min = scan_data.angle_min
    angle_max = scan_data.angle_max
    angle_increment = scan_data.angle_increment
    for i in range(len(ranges)):
        x = ranges[i] * math.sin(angle_min + i * angle_increment)
        y = ranges[i] * math.cos(angle_min + i * angle_increment)
        laser_points.append([x, y])
    return np.asarray(laser_points)

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