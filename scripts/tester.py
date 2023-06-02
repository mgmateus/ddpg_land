#! /usr/bin/env python3

from torch import float64, mean
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
MAX_STEPS= 2000
MAX_EPS= 1000
BUFFER_SIZE= 50000
BATCH_SIZE= 512

EPS_ = 1800
EPS= 0
test = 0
exemple = 4
max_eps_test = 10
IS_FIRST_EPS= False

if __name__ == "__main__":
    
    rospy.init_node("land", anonymous=False)

    env = Environment(SPACE_STATE_DIM, SPACE_ACTION_DIM, MAX_STEPS)
    agent = DDPGagent(SPACE_STATE_DIM, SPACE_ACTION_DIM, buffer_size=BUFFER_SIZE)

    rospy.Subscriber("/airsim_node/origin_geo_point", GPSYaw, env.drone_1.call_origin_geo_point)
    rospy.Subscriber("/airsim_node/drone_1/gps/gps", NavSatFix, env.drone_1.call_gps)
    rospy.Subscriber("/vision", ProcessedImg, env.drone_1.call_vision)

    rospy.Subscriber("/airsim_node/drone_1/odom_local_ned", Odometry, env.drone_1.call_odometry)


    if not IS_FIRST_EPS:
        
        agent.load_models(EPS_)
        rospy.logwarn("Model carregado %s", EPS_)

    time.sleep(2)
    eps = EPS
    done_ = True
    while eps < (MAX_EPS - EPS) or test < 100:

        rospy.logwarn("Epsiodeo atual: %s", eps)
        rospy.logwarn("Teste atual: %s", test)

        done = False
        
        state = env.reset(reset_target=True, eps_=eps)

        rewards_current_episode = 0.
        mean_rewards_current_episode = []

        if eps % 10 == 0:
            test += 1
            t_0 = rospy.Time.now().secs
        
        for step in range(MAX_STEPS):
            state = np.float32(state)
            action = agent.get_action(state)

            new_state, reward, done = env.step(action) 
            rewards_current_episode += reward
            mean_rewards_current_episode.append(reward)

            new_state = np.float32(new_state)
            
            if reward >= 200.:
                rospy.logwarn("--------- Maximum Reward ----------")
                done_ = True
                for _ in range(3):
                    agent.memory.push(state, action, reward, new_state, done)
            else:
                agent.memory.push(state, action, reward, new_state, done)
                done_ = False

            state = copy.deepcopy(new_state)   
            if done or step == MAX_STEPS-1:
                rospy.logwarn("Reward per ep: %s", str(rewards_current_episode))
                rospy.logwarn("Break step: %s", str(step))
            
                break
            rospy.Rate(10).sleep()
        
        if done_:
            eps += max_eps_test
            max_eps_test = 10
            break
        else:
            eps += 1
            max_eps_test -= 1

        agent.save_test(test, eps%10, rewards_current_episode, rospy.Time.now().secs-t_0, state[2:])
    env.save_odom(exemple)
    print("Complete Test")
