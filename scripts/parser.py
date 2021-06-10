from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from visualization_msgs.msg import Marker
import numpy as np
import math


#def laser_parser(scan_data, state):
#    #Assembles all points detected by LiDAR in an array
#    #Each point described by x,y coordinates
#    #These coordinates are from the perspective of the car
#    #These two callbacks are called by ROS to update the state
#    laser_points = []
#    ranges = scan_data.ranges
#    angle_min = scan_data.angle_min
#    angle_max = scan_data.angle_max
#    angle_increment = scan_data.angle_increment
#    for i in range(len(ranges)):
#        x = ranges[i] * math.sin(angle_min + i * angle_increment)
#        y = ranges[i] * math.cos(angle_min + i * angle_increment)
#        laser_points.append([x, y])
#    state.cur_points = np.asarray(laser_points)

def laser_parser(scan_data, state):
    state.line_scan = np.asarray(scan_data.ranges)

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
    cur_state = {
        'odom': cur_state,
        'laser_scan': main_state.line_scan
    }
    return cur_state

def publish_steering_prob(steering_prob, st_display_pub):
    choice = steering_prob.argmax()
    y_offset = -0.2
    width = 0.2
    for i in range(steering_prob.shape):
        marker_msg = Marker()
        marker_msg.header.frame_id = "ego_racecar"
        marker_msg.header.stamp = rospy.Time.now()
        marker_msg.ns = "st_probs"
        marker_msg.id = i
        marker_msg.type = 1
        marker_msg.action = 0
        marker_msg.pose.position.x = width((steering_prob/2)-i+0.5)
        marker_msg.pose.position.y = y_offset
        marker_msg.pose.position.z = 0
        marker_msg.pose.orientation.x = 0.0
        marker_msg.pose.orientation.y = 0.0
        marker_msg.pose.orientation.z = 0.0
        marker_msg.scale.x = width
        marker_msg.scale.y = steering_prob(i)
        marker_msg.scale.z = 0.1
        marker_msg.color.a = 1.0
        if(i==choice):
            marker_msg.color.r = 0
            marker_msg.color.g = 1.0
            marker_msg.color.b = 0
        else:
            marker_msg.color.r = 1.0
            marker_msg.color.g = 0
            marker_msg.color.b = 1.0
        st_display_pub.publish(marker_msg)