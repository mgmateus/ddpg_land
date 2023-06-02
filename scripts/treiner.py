#! /usr/bin/env python3

import rospy
import numpy as np
import copy
import time

from std_msgs.msg import *
from sensor_msgs.msg import *
from nav_msgs.msg import *
from airsim_ros_pkgs.msg import *
from rl_pkg.msg import *

from ddpg import DDPGagent
from ou_noise import OUNoise
from replay_buffer import ReplayBuffer
from environment import Environment

from airsim import Pose

from uav import Uav


SPACE_STATE_DIM= 4
SPACE_ACTION_DIM= 2
MAX_STEPS= 1000
MAX_EPS= 2001
BUFFER_SIZE= 50000
BATCH_SIZE= 512

EPS_ = 1780
EPS= 1780
IS_FIRST_EPS= False

if __name__ == "__main__":
    
    rospy.init_node("land", anonymous=False)

    env = Environment(SPACE_STATE_DIM, SPACE_ACTION_DIM, MAX_STEPS)
    agent = DDPGagent(SPACE_STATE_DIM, SPACE_ACTION_DIM, buffer_size=BUFFER_SIZE)
    noise = OUNoise(SPACE_ACTION_DIM)

    rospy.Subscriber("/airsim_node/origin_geo_point", GPSYaw, env.drone_1.call_origin_geo_point)
    rospy.Subscriber("/airsim_node/drone_1/gps/gps", NavSatFix, env.drone_1.call_gps)
    rospy.Subscriber("/vision", ProcessedImg, env.drone_1.call_vision)

    if not IS_FIRST_EPS:
        
        agent.load_models(EPS_)
        rospy.logwarn("Model carregado %s", EPS_)

    time.sleep(3)
    for eps in range(EPS, MAX_EPS):
        rospy.logwarn("Epsiodeo atual: %s", eps)

        done = False
        state = env.reset(reset_target=True, eps_= eps)
        noise.reset()
        rewards_current_episode = 0.
        
        
        for step in range(MAX_STEPS):
            state = np.float32(state)
            action = agent.get_action(state)
            
            
            N = copy.deepcopy(noise.get_noise(t=step))        
            action[0] = np.clip(action[0] + (N[0]*.75), -1, 1)
            action[1] = np.clip(action[1] + (N[1]*.75), -1, 1)
                        
                
            new_state, reward, done = env.step(action) 
            rewards_current_episode += reward
            new_state = np.float32(new_state)

            if not eps%10 == 0 or not len(agent.memory) >= SPACE_ACTION_DIM*BATCH_SIZE:
                if reward >= 200.:
                    rospy.logwarn("--------- Maximum Reward ----------")
                    
                    for _ in range(3):
                        agent.memory.push(state, action, reward, new_state, done)
                else:
                    agent.memory.push(state, action, reward, new_state, done)

            if len(agent.memory) > SPACE_ACTION_DIM*BATCH_SIZE and not eps%10 == 0:
                rospy.logwarn("--------- UPDATE ----------")
                agent.update(BATCH_SIZE) # ----- 1 -----

            state = copy.deepcopy(new_state)   

            agent.save_state(eps, step, reward, state)

            if done or step == MAX_STEPS-1:
                rospy.logwarn("Reward per ep: %s", str(rewards_current_episode))
                rospy.logwarn("Break step: %s", str(step))
                rospy.logwarn("Sigma: %s", str(noise.sigma))
            
                break

            
            rospy.Rate(10).sleep()

        agent.save_models(eps)
        agent.save_rewards(eps, rewards_current_episode)
    
