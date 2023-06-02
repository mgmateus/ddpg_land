#! /usr/bin/env python3

from pydoc import doc
import time
import numpy as np
from cmath import *
from math import *
from pymap3d import *
import random
import os
import csv


from std_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from airsim_ros_pkgs.msg import *
from rl_pkg.msg import *

from airsim_ros_pkgs.srv import *


from uav import Uav
from ddpg import DDPGagent
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer

import airsim



class Environment:
    def __init__(self, space_state_dim, space_action_dim, max_steps):
        self.drone_1 = Uav()
        
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()


        self.observation_space = np.zeros(shape=(space_state_dim,))
        self.space_action_dim = space_action_dim
        self.past_action_space = np.zeros(shape=(space_action_dim,))
        
        self.distance2land_t_1 = None
        self.altitude2land_t_1 = None
        self.altitude2land_t_0 = None

        self.n_steps = 0
        self.max_steps = max_steps

        self.client.reset()
        time.sleep(2)
        while True:
            try:
                self.drone_1.reset()
                #time.sleep(5)
                break
            except:
                pass
        self.client.startRecording()

    def reset_simulation(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def reset_target(self):
        offset_x = random.randint(-4,4)
        offset_y = random.randint(-4,4)
        offset_w = random.random() * random.randint(-1,1)
        offset_z = random.random() * random.randint(-1,1)

        pose = airsim.Pose()
        pose.orientation.w_val= offset_w
        pose.orientation.x_val= float()
        pose.orientation.y_val= float()
        pose.orientation.z_val= offset_z
    
        pose.position.x_val = self.client.simGetVehiclePose("drone_1").position.x_val + offset_x
        pose.position.y_val = self.client.simGetVehiclePose("drone_1").position.y_val + offset_y
        pose.position.z_val = 11.396

        self.client.simSetObjectPose("Boat2_5", pose)

    def reset(self, reset_target= False, eps_= 0, reset_now= False):
        while True:
            try:
                self.reset_simulation()
                if (reset_target and eps_ == 0) or (reset_target and reset_now) or (reset_target and eps_ % 10 == 0):
                    self.reset_target()
                        
                self.n_steps = 0
                _ = self.drone_1.getState(np.zeros(shape=(self.space_action_dim,)))
                break
            except:
                pass
        
        return self.observation_space


    def setReward(self, done):
        if done: 
            self.drone_1.vel.twist.linear.x = 0
            self.drone_1.vel.twist.linear.y = 0
            self.drone_1.vel_pub.publish(self.drone_1.vel)
            pose = airsim.Pose()
            pose.orientation.w_val= float()
            pose.orientation.x_val= float()
            pose.orientation.y_val= float()
            pose.orientation.z_val= float()
            pose.position.x_val = self.client.simGetVehiclePose("drone_1").position.x_val
            pose.position.y_val = self.client.simGetVehiclePose("drone_1").position.y_val
            pose.position.z_val = 10.77623558044434
    
            self.client.simSetVehiclePose(pose, True, "drone_1")

            time.sleep(3)
            _ = self.drone_1.getState(np.zeros(shape=(self.space_action_dim,)))
            #self.reset_simulation()
            return 250

        else:
            
            r = 0
            try:
                
                if self.distance2land_t_1 is None:
                    r = 0    

                if self.drone_1.distance2land == 1000000.0 and self.n_steps > 0:
                    _ = self.drone_1.getState(np.zeros(shape=(self.space_action_dim,)))
                    self.reset_simulation()
                    r= -10
                    
            
                elif (self.drone_1.distance2land - self.distance2land_t_1) < 0:
                    r= .1

                else:
                    r= 0
            except:
                r= 0

            self.distance2land_t_1 = self.drone_1.distance2land
            return r 
            

    def step(self, action):
        
        self.observation_space, done = self.drone_1.getState(action)
        reward = self.setReward(done)
        self.n_steps += 1
        
        if self.n_steps < self.max_steps:
            return self.observation_space, reward, done
        else:
            return self.observation_space, 0.0, True

    
    def save_odom(self, test):
        dirPath = os.path.dirname(os.path.realpath(__file__))
        logPath = dirPath.replace("/scripts", "/log")
        print(logPath +"/test/" + str(test) + "_odom.csv")
        f = open(logPath +"/test/" + str(test) + "_odom.csv", "a", encoding='utf-8', newline='')
        w = csv.writer(f)
        for pose in self.drone_1.odm_nav:            
            w.writerow([pose[0], pose[1], pose[2]])
        f.close() 
        

        

