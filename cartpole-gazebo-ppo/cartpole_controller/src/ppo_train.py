#!/usr/bin/python3
import gym
import numpy as np
import tensorflow as tf
from policy_net import Policy_net
from ppo import PPOTrain

import roslib
import rospy
import random
import time
import math
import csv
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetModelConfiguration

from control_msgs.msg import JointControllerState
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import LinkStates
from gazebo_msgs.srv import SetLinkState
from gazebo_msgs.msg import LinkState
from std_msgs.msg import Float64
from std_msgs.msg import String
from sensor_msgs.msg import Joy

ENV_NAME = 'Cartpole_v0'
ITERATION = 2000
GAMMA = 0.95

pubCartPosition = rospy.Publisher('/stand_cart_position_controller/command', Float64, queue_size=1)
pubJointStates = rospy.Publisher('/joint_states', JointState, queue_size=1)

reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
reset_joints = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
set_link = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)

fall = 0


rospy.init_node('cartpole_control_script')
rate = rospy.Rate(120)

class RobotState(object):
    def __init__(self):
        self.cart_x = 0.0
        self.cart_x_dot = 0.0
        self.pole_theta = 0.0
        self.pole_theta_dot = 0.0
        self.robot_state = [self.cart_x, self.cart_x_dot, self.pole_theta, self.pole_theta_dot]
        
        self.data = None
        self.latest_reward = 0.0
        self.fall = 0

        self.theta_threshold = 0.20943951023
        self.x_threshold = 0.4

        self.current_vel = 0.0
        self.done = False


robot_state = RobotState()


def reset():
    rospy.wait_for_service('/gazebo/reset_world')

    try:
        reset_world()
    except (rospy.ServiceException) as e:
        print ('reset_world failed!')


        # rospy.wait_for_service('/gazebo/reset_world')
    rospy.wait_for_service('/gazebo/set_model_configuration')

    try:
        #reset_proxy.call()
        # reset_world()
        reset_joints("cartpole", "robot_description", ["stand_cart", "cart_pole"], [0.0, 0.0])


    except (rospy.ServiceException) as e:
        print ('/gazebo/reset_joints service call failed')

    rospy.wait_for_service('/gazebo/pause_physics')
    try:
        pause()
    except (rospy.ServiceException) as e:
        print ('rospause failed!')

    # rospy.wait_for_service('/gazebo/unpause_physics')
    
    # try:
    #     unpause()
    # except (rospy.ServiceException) as e:
    #     print "/gazebo/pause_physics service call failed"

    set_robot_state()
    robot_state.current_vel = 0
    print ('called reset()')

def set_robot_state():
    robot_state.robot_state = [robot_state.cart_x, robot_state.cart_x_dot, robot_state.pole_theta, robot_state.pole_theta_dot]

def take_action(action):
    rospy.wait_for_service('/gazebo/unpause_physics')
    
    try:
        unpause()
    except (rospy.ServiceException) as e:
        print ('/gazebo/pause_physics service call failed')

    
    if action == 1:
        robot_state.current_vel = robot_state.current_vel + 0.05
    else:
        robot_state.current_vel = robot_state.current_vel - 0.05


    # print "publish : ", robot_state.current_vel
    pubCartPosition.publish(robot_state.current_vel)
    
    reward = 1

    # ['cart_pole', 'stand_cart']
    if robot_state.data==None:
        while robot_state.data is None:
            try:
                robot_state.data = rospy.wait_for_message('/joint_states', JointState, timeout=5)
            except:
                print ('Error getting /joint_states data.')

    set_robot_state()

    if robot_state.cart_x < -robot_state.x_threshold or robot_state.cart_x > robot_state.x_threshold or robot_state.pole_theta > robot_state.theta_threshold \
    or robot_state.pole_theta < -robot_state.theta_threshold:
       
        robot_state.done = True
        reward = 1

    else:
        reward = 1

    # rate.sleep()

    return reward, robot_state.done


def callbackJointStates(data):
    if len(data.velocity) > 0:
        robot_state.cart_x_dot = data.velocity[1]
        robot_state.pole_theta_dot = data.velocity[0]
    else:
        robot_state.cart_x_dot = 0.0
        robot_state.pole_theta_dot = 0.0
    robot_state.cart_x = data.position[1]
    robot_state.pole_theta = data.position[0]

    set_robot_state()

    # print ('DATA :'), data


def listener():
    print ('listener')
    rospy.Subscriber("/joint_states", JointState, callbackJointStates)
    


def main():
    listener()
    # env = gym.make('CartPole-v0')
    # env.seed(0)
    ob_space = 4
    Policy = Policy_net('policy')
    Old_Policy = Policy_net('old_policy')
    PPO = PPOTrain(Policy, Old_Policy, gamma=GAMMA)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./log/train', sess.graph)
        sess.run(tf.global_variables_initializer())
        reset()
        obs = robot_state.robot_state
        reward = 0
        success_num = 0

        for iteration in range(ITERATION):  # episode
            observations = []
            actions = []
            v_preds = []
            rewards = []
            run_policy_steps = 0
            while True:  # run policy RUN_POLICY_STEPS which is much less than episode length
                run_policy_steps += 1
                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)
                print('act: ',act, 'v_pred: ',v_pred )
                act = np.asscalar(act)
                v_pred = np.asscalar(v_pred)

                observations.append(obs)
                actions.append(act)
                v_preds.append(v_pred)
                rewards.append(reward)

                reward, done = take_action(act)
                time.sleep(0.25)
                next_obs = robot_state.robot_state

                if done:
                    v_preds_next = v_preds[1:] + [0]  # next state of terminate state has 0 state value
                    reset()
                    obs = robot_state.robot_state
                    reward = -1
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)

            if sum(rewards) >= 195:
                success_num += 1
                render = True
                if success_num >= 100:
                    saver.save(sess, './model/model.ckpt')
                    print('Clear!! Model saved.')
                    break
            else:
                success_num = 0

            gaes = PPO.get_gaes(rewards=rewards, v_preds=v_preds, v_preds_next=v_preds_next)

            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, [len(observations), 4])
            actions = np.array(actions).astype(dtype=np.int32)
            rewards = np.array(rewards).astype(dtype=np.float32)
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            gaes = np.array(gaes).astype(dtype=np.float32)
            gaes = (gaes - gaes.mean()) 
            print('gaes', gaes)

            PPO.assign_policy_parameters()

            inp = [observations, actions, rewards, v_preds_next, gaes]

            # train
            for epoch in range(4):
                sample_indices = np.random.randint(low=0, high=observations.shape[0], size=64)  # indices are in [low, high)
                sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                PPO.train(obs=sampled_inp[0],
                          actions=sampled_inp[1],
                          rewards=sampled_inp[2],
                          v_preds_next=sampled_inp[3],
                          gaes=sampled_inp[4])

            summary = PPO.get_summary(obs=inp[0],
                                      actions=inp[1],
                                      rewards=inp[2],
                                      v_preds_next=inp[3],
                                      gaes=inp[4])[0]

            writer.add_summary(summary, iteration)
        writer.close()


if __name__ == '__main__':
    main()
