from multiprocessing.connection import wait
from pydoc import doc
import time
import rospy
import rosnode
import roslaunch

from cmath import *
from math import *
from pymap3d import *
import numpy as np
import os


from std_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from airsim_ros_pkgs.msg import *
from rl_pkg.msg import *
from airsim_ros_pkgs.srv import *


class Uav:
    def __init__(self):
        #Publishers
        self.home_geo_point_pub = rospy.Publisher("/airsim_node/home_geo_point", GPSYaw, queue_size=50)
        self.vel_pub = rospy.Publisher("/airsim_node/drone_1/vel_cmd_world_frame", VelCmd, queue_size=1)

        #Parametros a receber
        self.gps = NavSatFix()
        self.vel = VelCmd()
        self.processedImg = ProcessedImg()
        self.odometry = Odometry()

        self.odm_nav = []
        
    #Callbacks
    def call_origin_geo_point(self, origin_geo_point_msg):
        self.home_geo_point_pub.publish(origin_geo_point_msg)

    def call_gps(self, gps_msg):
        self.gps = gps_msg
        self.altitude2land = self.gps.altitude

    def call_vision(self, processedImg_msg):
        self.processedImg = processedImg_msg
        if self.processedImg.dxPixel == 1000000.0 or self.processedImg.dyPixel == 1000000.0:
            self.distance2land =  1000000.0
            self.dx = self.processedImg.dxPixel
            self.dy = self.processedImg.dyPixel
        else:
            self.dx = self.processedImg.dxPixel
            self.dy = self.processedImg.dyPixel
            self.distance2land =  hypot(self.dx, self.dy)

    def call_odometry(self, odom_msg):
        self.odometry = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z]

    #Requests srv
    def takeOff(self):
        try:
            service = rospy.ServiceProxy("/airsim_node/drone_1/takeoff", Takeoff)
            rospy.wait_for_service("/airsim_node/drone_1/takeoff")

            service()

        except rospy.ServiceException as e:
            print ('Service call failed: %s' % e)

    def land(self):
        try:
            service = rospy.ServiceProxy("/airsim_node/drone_1/land", Land)
            rospy.wait_for_service("/airsim_node/drone_1/land")

            service()

        except rospy.ServiceException as e:
            print ('Service call failed: %s' % e)

    #Movement
    def up_down(self, h):
        self.vel.twist.linear.x = 0
        self.vel.twist.linear.y = 0
        gps_msg = self.gps
        target = gps_msg.altitude + h
        delta_h = abs(gps_msg.altitude - target)
        error = target * 0.3
        vel_cmd = -1 if h > 0 else 1
        
        while delta_h >= error:
            self.vel.twist.linear.z = vel_cmd
            self.vel_pub.publish(self.vel)         
            gps_msg = self.gps
            delta_h = abs(gps_msg.altitude - target)

        self.vel.twist.linear.z = 0
        self.altitude2land = self.gps.altitude 

    
    def move(self, action):
        self.vel.twist.linear.x = action[0]
        self.vel.twist.linear.y = action[1]
        self.vel_pub.publish(self.vel)

    #Utils
    def getState(self, action):
        self.odm_nav += [self.odometry]
        self.vel.twist.linear.x = np.clip(action[0], -.25, .25)
        self.vel.twist.linear.y = np.clip(action[1], -.25, .25)
        #self.vel.twist.linear.z = action[2]
        self.vel_pub.publish(self.vel)
        try:
            done = True if self.distance2land <= 10 else False
            if done:
                self.up_down(-9)

            return [self.dx, self.dy, action[0], action[1]], done
            
        except:
            done = False
            self.dx = 1000000.0
            self.dy = 1000000.0
            return [self.dx, self.dy, action[0], action[1]], done

        
        
    def reset_vision(self):    
        _ = rosnode.kill_nodes(["vision"])
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [os.path.dirname(os.path.realpath(__file__)).replace("/scripts", "/launch/vision.launch")])
        launch.start()

    def reset(self):
        self.gps = NavSatFix()
        self.vel = VelCmd()
        self.processedImg = ProcessedImg()

        self.reset_vision()

        

        

    
