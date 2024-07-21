# Cassie robot training environment
# Author: XIE Yiming 510001869
#%%
import dill
import time
import itertools
import mediapy as media
import matplotlib.pyplot as plt
from datetime import datetime
import functools
from IPython.display import HTML
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from etils import epath
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco.viewer
from mujoco import mjx

class CassieEnv(PipelineEnv):
    """Trains a Cassie robot to walk forward."""

#------------------------------------------------------------------------------------
#                                   Initialization    
    
    def __init__(
            self,
            obs_noise: float = 0.05,
            disturbance_vel: float = 0.05,
            contact_limit: float = 0.021,
            done_limit: float = 0.6,
            timestep: float = 0.025,
            action_scale: float = 0.5,
            **kwargs,
    ):
        # load cassie model
        path = '/home/icepomelo/Code/mujoco_menagerie/agility_cassie/scene_mjx.xml'
        model = mujoco.MjModel.from_xml_path(path)
        sys = mjcf.load_model(model)
        
        # set how many simulation step to take one action
        # here use 50, means after 50 simulation step, the action will be updated
        # simulation timestep = 0.0005s
        n_frames = kwargs.pop('n_frames', 50)

        # initialize the superclass
        super().__init__(
            sys=sys,
            backend='mjx',
            n_frames=n_frames
        )

        # store observation noise coefficient
        self.obs_noise = obs_noise
        # store disturbance velocity
        self.disturb_vel = disturbance_vel
        # store contact limit
        self.contact_limit = contact_limit
        # store done limit
        self.done_limit = done_limit
        # store timestep
        self.timestep = timestep
        # store action scale
        self.action_scale = action_scale

        # store site position for contact detection
        feet_site = ["left_feet_back","left_feet_front","right_feet_back","right_feet_front"]
        feet_site_id = [mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f) for f in feet_site]
        self.feet_site_id = jp.array(feet_site_id)

        # store actuator range
        self.action_range = sys.actuator_ctrlrange
        # store initial joint position
        self.inital_qpos = jp.array(sys.mj_model.keyframe('home').qpos)
        # store default joint angle in joint space
        self.default_jnt_angle = jp.concatenate([
            sys.mj_model.keyframe('home').qpos[0:3],
            sys.mj_model.keyframe('home').qpos[7:10],
            sys.mj_model.keyframe('home').qpos[14:17],
            sys.mj_model.keyframe('home').qpos[20].reshape(-1),
            sys.mj_model.keyframe('home').qpos[21:24],
            sys.mj_model.keyframe('home').qpos[28:31],
            sys.mj_model.keyframe('home').qpos[34].reshape(-1)
        ])
        # store motor joint angle
        self.motor_angle = jp.concatenate([
            sys.mj_model.keyframe('home').qpos[7].reshape(-1),
            sys.mj_model.keyframe('home').qpos[8].reshape(-1),
            sys.mj_model.keyframe('home').qpos[9].reshape(-1),
            sys.mj_model.keyframe('home').qpos[14].reshape(-1),
            sys.mj_model.keyframe('home').qpos[20].reshape(-1),
            sys.mj_model.keyframe('home').qpos[21].reshape(-1),
            sys.mj_model.keyframe('home').qpos[22].reshape(-1),
            sys.mj_model.keyframe('home').qpos[23].reshape(-1),
            sys.mj_model.keyframe('home').qpos[28].reshape(-1),
            sys.mj_model.keyframe('home').qpos[34].reshape(-1)
        ])
        # store the joint limit
        self.joint_limit = jp.array([
            model.jnt_range[1],
            model.jnt_range[2],
            model.jnt_range[3],
            model.jnt_range[5],
            model.jnt_range[6],
            model.jnt_range[7],
            model.jnt_range[9],
            model.jnt_range[11],
            model.jnt_range[12],
            model.jnt_range[13],
            model.jnt_range[14],
            model.jnt_range[16],
            model.jnt_range[17],
            model.jnt_range[18],
            model.jnt_range[20],
            model.jnt_range[22]
        ])
        # store default standing force
        self.standing = sys.mj_model.keyframe('home').ctrl
        # store size of joint space
        self.jnt_size = len(self.default_jnt_angle)
        # store the degree of freedom
        self.nv = sys.nv
        # store the number of actuators
        self.nu = sys.nu

        # store information from body list
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cassie-pelvis') 
        self.left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left-foot') 
        self.right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'right-foot') 
        self.foot_id = jp.array([self.left_foot_id,self.right_foot_id])


#------------------------------------------------------------------------------------
#                                      Step
    
    def step(
            self, 
            state: State, 
            action: jax.Array
    ) -> State:
        """
        Take a step in the environment.

        input: 
            state: the current state of the environment
            action: the action to take in the environment

        output:
            state: the new state of the environment
        """

        # spilt random key generator for different use
        rng, ctl_rng, disturb_rng = jax.random.split(state.info['Random generator'],3)

    #____________________________________________________________________________
    # Disturbanace

    # The disturbance will present as a random velocity to pelvis along xy plane between
    # a predefined interval. Add disturbance to the robots to increase robustness

        # # set disturbance interval
        # disturb_interval = 40       # every 20 steps, apply a disturbance

        # # generate the disturbance velocity direction and represent it in xy plane
        # # when the current step is a multiple of disturb_interval, apply disturbance

        # # generate a random angle for disturbance between 0 to 2pi
        # disturb_angle = jax.random.uniform(disturb_rng, maxval = 2 * jp.pi)

        # # calculate the disturbance velocity and set it to xy plane
        # disturb = jp.array([jp.cos(disturb_angle), jp.sin(disturb_angle)]) * self.disturb_vel
        
        # # check if the current step is a multiple of disturb_interval
        # disturb *= jp.mod(state.info['Step total'], disturb_interval) == 0
        
        # # apply disturbance to the robot by replace the first 2 velocity in qvel
        # # it represent the pelvis velocity in xy plane
        # qvel = state.pipeline_state.qvel.at[:2].set(disturb + state.pipeline_state.qvel[:2])
        
        # # update the state
        # state = state.tree_replace({'pipeline_state.qvel': qvel})
    
    #____________________________________________________________________________
    # Action scale and step

    # Apply the action to the state and step it forward. The action should be scaled to its
    # actuators range

        # scale the action to actuators range
        action = self._action_scale(action)

        # action the motor to its default pose
        motor_action = self.standing + action * self.action_scale

        # update the state
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_action)

    #____________________________________________________________________________
    # State information extraction

    # Extract the information:
    # body_pos: position or orientation of the robot
    # body_vel: derivative of x, velocity or angular velocity of the robot
    # joint_angle: joint angle of the robot (only important joint)
    # limited_angle: joint angle of the robot with limit
    # joint_velocity: joint velocity of the robot
        body_pos = pipeline_state.x
        body_vel = pipeline_state.xd
        joint_angle = jp.concatenate([
            pipeline_state.q[0:3],
            pipeline_state.q[7:10],
            pipeline_state.q[14:17],
            pipeline_state.q[20].reshape(-1),
            pipeline_state.q[21:24],
            pipeline_state.q[28:31],
            pipeline_state.q[34].reshape(-1)
        ])
        limited_angle = jp.concatenate([
            pipeline_state.q[7:10],
            pipeline_state.q[14:17],
            pipeline_state.q[18].reshape(-1),
            pipeline_state.q[20].reshape(-1),
            pipeline_state.q[21:24],
            pipeline_state.q[28:31],
            pipeline_state.q[32].reshape(-1),
            pipeline_state.q[34].reshape(-1)
        ])
        joint_velocity = jp.concatenate([
            pipeline_state.qd[0:3],
            pipeline_state.qd[6:9],
            pipeline_state.qd[12:15],
            pipeline_state.qd[18].reshape(-1),
            pipeline_state.qd[19:22],
            pipeline_state.qd[25:28],
            pipeline_state.qd[31].reshape(-1)
        ])

    #____________________________________________________________________________
    # Observation

    # Get the observation of the environment
        obs = self._get_obs(pipeline_state, state.info, joint_angle, joint_velocity)

    #____________________________________________________________________________
    # foot contact and feet air time

    # determine the current status of the feet and obtain the air time

        # obtain the foot body position
        foot_pos = pipeline_state.site_xpos[self.feet_site_id]
        foot_pos = foot_pos[:,2]

        left_foot_pos = foot_pos[0:2]
        right_foot_pos = foot_pos[2:4]

        left_contact = jp.all(left_foot_pos < self.contact_limit)
        right_contact = jp.all(right_foot_pos < self.contact_limit)

        # check if the feet are in contact with the ground
        contact = jp.array([left_contact, right_contact])

        # combine the current contact status with the last contact stauts
        # to have a smoother transition
        contact_filt = contact 

        # identify if it is the first contact after in the air
        first_contact = (state.info['Feet air time']>0) * contact_filt

        # update the feet air time
        state.info['Feet air time'] += self.timestep

    #____________________________________________________________________________
    # Termination of epoisode
    
    # check if the episode should be terminated by checking if it has fallen

        # if the pelvis is too close to the ground, it indicates a fall
        # check the pelvis position in z axis
        done = body_pos.pos[self.pelvis_id-1,2] < self.done_limit

    #____________________________________________________________________________
    # Desired walking orientation
    
    # calculate the forward orientation of the robot
        orientation = self._calculate_forward_orientation(state.info['Control commands'])

    #____________________________________________________________________________
    # Reward

    # Built reward functions for robot to walking around

        rewards = {
            'tracking linear velocity reward':(
                2*self._linear_velocity_tracking(state.info['Control commands'],body_vel)),

            # 'tracking angular velocity reward':(
            #     1.5*self._angular_velocity_tracking(state.info['Control commands'],body_vel)),

            'z axis velocity penalty':(
                -1*self._linear_velocity_penalty(body_vel)),

            'x-y plane angular velocity penalty':(
                -0.5*self._angular_velocity_penalty(body_vel)),

            'joint torque penalty':(
                -0.00005*self._joint_torque_penalty(pipeline_state.qfrc_actuator)),

            'action rate penalty':(
                -0.1*self._action_rate_penalty(action,state.info['Last action'])),

            'offset when standing still penalty':(
                -1*self._motion_standing_still_penalty(state.info['Control commands'],joint_angle)),

            # 'alive reward':(
            #     3*self._alive_reward(done,state.info['Step'])),

            'feet air time reward':(
                8*self._feet_air_time_reward(state.info['Feet air time'],first_contact,state.info['Control commands'])),

            'Single feet on ground reward':(
                0.3*self._single_leg_on_ground_reward(contact,state.info['Control commands'])),

            'early termination penalty':(
                -1*self._early_termination_penalty(done,state.info['Step total'])),
            
            # 'Both leg in the air penalty':(
            #     -2*self._both_foot_air_penalty(contact)),

            'joint limit penalty':(
                -2*self._joint_limit_penalty(limited_angle)),

            'orientation penalty':(
                -1*self._orientation_penalty(body_pos)),

            'feet position penalty':(
                -2*self._feet_position_penalty(body_pos)),

            'z position penalty':(
                -2*self._z_position_penalty(body_pos)),

            'orientation walking penalty':(
                -3*self._orientation_walking_penalty(orientation,body_pos))
        }

        # calculate the total reward
        reward = sum(rewards.values())
    
    #____________________________________________________________________________
    # State information update
    
    # update state.info to track the training progress and gather useful infomation

        # update the state information
        state.info['State'] = pipeline_state
        state.info['Random generator'] = rng
        state.info['Last action'] = action
        state.info['Last pelvis position'] = body_pos.pos[self.pelvis_id-1]
        state.info['Last pelvis velocity'] = body_vel.vel[self.pelvis_id-1]
        state.info['Last joint angle'] = joint_angle
        state.info['Last feet contact'] = contact
        state.info['Feet air time'] *= ~contact_filt
        state.info['Reward'] = reward
        # state.info['Reward dict'] = rewards
        state.info['Step'] += 1
        state.info['Step total'] += 1
        state.info['Total distance'] = math.normalize(body_pos.pos[self.pelvis_id-1][:2])[1]

        # update the control commands when more than 500 timestep has achieved
        state.info['Control commands'] = jp.where(
        # condition: step>500
            state.info['Step'] > 500,
        # if true
            self.control_commands(ctl_rng),
        # if false
            state.info['Control commands']
        )
    
        # reset the step counter when the episode is terminated or reached 500 steps
        state.info['Step'] = jp.where(
        # condition: done or step>500
            done | (state.info['Step'] > 500),
        # if true
            0,
        # if false
            state.info['Step']
        )

        # log total displacement as a proxy metric
        state.metrics['Total distance'] = state.info['Total distance']
        # update the reward into proxy metric
        state.metrics['reward'] = reward
        # convert termination flag
        done = jp.float32(done)
    #____________________________________________________________________________
    # Wrap the state
        state = state.replace(
            pipeline_state = pipeline_state,
            obs = obs,
            reward = reward,
            done = done,
        )

        return state

#------------------------------------------------------------------------------------
#                                      Reset
    
    def reset(
            self,
            rng:jax.Array,
    ) -> State:
        """
        Reset the environment to the initial state.
        
        input: 
            rng: random key generator from Jax
        output: 
            state: the initial state of the environment
        """
        # split the random key
        rng, key = jax.random.split(rng)

        # initialize the pipeline state with initial position and zero velocity
        pipeline_state = self.pipeline_init(self.inital_qpos,jp.zeros(self.nv))
        # create state information disctrionary to store state information
        state_info = {

            'State': pipeline_state,
            'Random generator': rng,
            'Control commands': self.control_commands(key),
            'Last action': jp.zeros(self.nu),
            'Last pelvis position': jp.zeros(3),
            'Last pelvis velocity': jp.zeros(3),
            'Last joint angle': jp.zeros(self.jnt_size),
            'Last feet contact': jp.zeros(2,dtype=bool),
            'Feet air time': jp.zeros(2),
            'Reward': 0.0,
            'Step': 0,
            'Step total': 0,
            # 'Reward dict': {}
            'Total distance': 0.0
        }
        # initialize reward and done for state
        reward, done = jp.zeros(2)

        # initialize additional metrics
        metrics = {'Total distance': 0.0,
                   'reward':0.0}
        
        joint_angle = jp.concatenate([
            pipeline_state.q[0:3],
            pipeline_state.q[7:10],
            pipeline_state.q[14:17],
            pipeline_state.q[20].reshape(-1),
            pipeline_state.q[21:24],
            pipeline_state.q[28:31],
            pipeline_state.q[34].reshape(-1)
        ])
        joint_velocity = jp.concatenate([
            pipeline_state.qd[0:3],
            pipeline_state.qd[6:9],
            pipeline_state.qd[12:15],
            pipeline_state.qd[18].reshape(-1),
            pipeline_state.qd[19:22],
            pipeline_state.qd[25:28],
            pipeline_state.qd[31].reshape(-1)
        ])

        obs = self._get_obs(pipeline_state, state_info, joint_angle, joint_velocity)

        # construct the state
        state= State(
            pipeline_state = pipeline_state,
            obs = obs,
            reward = reward,
            done = done,
            metrics = metrics,
            info = state_info
        )

        return state
#------------------------------------------------------------------------------------
#                                   Observation
    def _get_obs(
            self,
            pipeline_state: State,
            state_info: dict[str,Any],
            joint_angle: jax.Array,
            joint_velocity: jax.Array
    ) -> jax.Array:
        """
        Get the observation of the environment.
        
        input: 
            pipeline_state: the state of the robot
            state_info: the information of the state
            joint_angle: the joint angle of the robot
            joint_velocity: the joint velocity of the robot

        output: 
            obs: the observation of the environment
        """
        # calculate the inverse of quaternion of pelvis
        inv_pelvis_rot = math.quat_inv(pipeline_state.x.rot[self.pelvis_id-1])

        # create observation vector
        obs = jp.concatenate([
            # pelvis linear velocity
            pipeline_state.xd.vel[self.pelvis_id-1]*2.0,

            # pelvis angular velocity
            pipeline_state.xd.ang[self.pelvis_id-1]*0.25,

            # direction of gravity relative to pelvis orientation
            math.rotate(jp.array([0,0,-1]),inv_pelvis_rot),

            # control command
            (state_info['Control commands'] * jp.array([2.0,2.0])).flatten(),

            # joint angle difference
            joint_angle - self.default_jnt_angle,

            # joint velocity
            joint_velocity * 0.05,

            # action
            state_info['Last action'],

            # feet position
            pipeline_state.xpos[self.foot_id].flatten()
            ])
        # clip the observation to prevent extreme values
        # add noise to encourage robustness
        obs = obs + self.obs_noise * jax.random.uniform(
            state_info['Random generator'], shape=obs.shape ,minval=-1, maxval=1)

        return obs

#------------------------------------------------------------------------------------
#                                 Control commands 
        
    def control_commands(
            self,
            rng:jax.Array,
    ) -> jax.Array:
        """
        Set the random velocity reference for the agent to track.
        
        input: 
            rng: random key generator from Jax

        output: 
            col_command: the random velocity reference for the agent to track
        
        """
        # set the random keys
        key1, key2, key3 = jax.random.split(rng,3)

        # set the control commands limitation
        velocity_x_limit = [0.0, 1.5]
        velocity_y_limit = [-0.5, 0.5]
        # angular_velocity_limit = [-0.5, 0.5]

        # set the random values for velocity
        velocity_x_command=jax.random.uniform(key1,shape=(1,),minval=velocity_x_limit[0],maxval=velocity_x_limit[1])
        velocity_y_command=jax.random.uniform(key2,shape=(1,),minval=velocity_y_limit[0],maxval=velocity_y_limit[1])    
        # angular_velocity_command=jax.random.uniform(key3,shape=(1,),minval=angular_velocity_limit[0],maxval=angular_velocity_limit[1])

        # combine the values
        col_command = jp.array([velocity_x_command[0],velocity_y_command[0]])
        return col_command

#------------------------------------------------------------------------------------
#                                   action scale
    def _action_scale(
            self, 
            action: jax.Array
    ) -> jax.Array:
        """
        Scale the action to the actuators range. The formula fron interval [-1,1] to [a,b] is:

        y = (x+1)*(b-a)/2 + a

        input: 
            action: the action to scale
        
        output:
            scaled_action: the scaled action
        """

        # define new interval range
        a = self.action_range[:,0]
        b = self.action_range[:,1]

        # scale the action
        scaled_action = (action+1)*(b-a)/2 + a
        
        return scaled_action
    
#------------------------------------------------------------------------------------
#                                   forward orientation checking
    def _calculate_forward_orientation(
            self,
            command:jax.Array,
            tolerance = 1e-7
    ) -> jax.Array:
        """
        Calculate the forward orientation of the robot

        input: 
            command: the control command of the robot

        output:
            orientation: the forward orientation of the robot
        """
        # get velocity from command
        velocity_x, velocity_y = command

        # calculate the angle in radians
        theta = jp.arctan2(velocity_y, velocity_x)  

        # calculate the cosine of the angle
        cos_theta = jp.cos(theta)  
        # calculate the sine of the angle
        sin_theta = jp.sin(theta)  

         # set small values to zero
        cos_theta = jp.where(jp.abs(cos_theta) < tolerance, 0, cos_theta)
        sin_theta = jp.where(jp.abs(sin_theta) < tolerance, 0, sin_theta)

        # create orientation vector
        orientation = jp.array([cos_theta,sin_theta])

        return orientation

#------------------------------------------------------------------------------------
#                                  Reward functions
    def _linear_velocity_tracking(self,
                                  command:jax.Array,
                                  body_vel:Motion) -> jax.Array:
        """
        reward term for tracking the reference linear velocity

        formula:
            q_linear = exp(-error^2/0.25)
        """

        # calculate the local velocity of robots from global velocity
        # local_vel = math.rotate(body_vel.vel[self.pelvis_id-1],math.quat_inv(body_pos.rot[self.pelvis_id-1]))

        # calculate the error
        linear_vel_err = jp.sum(jp.square(command[:2] - body_vel.vel[self.pelvis_id-1][:2]))

        # calculate the reward term
        reward = jp.exp(-linear_vel_err/0.25)

        return reward
    
    def _angular_velocity_tracking(self,
                                   command:jax.Array,
                                   body_vel:Motion) -> jax.Array:
        """
        reward term for tracking the reference angular velocity

        formula:
            q_angular = exp(-error^2/0.25)
        """
        
        # calculate the error
        angular_vel_err = jp.sum(jp.square(command[2] - body_vel.ang[self.pelvis_id-1][2]))

        # calculate the reward term
        reward = jp.exp(-angular_vel_err/0.25)

        return reward
    
    def _linear_velocity_penalty(self,
                                 body_vel:Motion) -> jax.Array:
        """
        penalty term for z axis velocity of pelvis

        provide steady walking and pace

        formula:
            q_z = -velocity_z^2
        """
        # calculate reward
        reward = jp.square(body_vel.vel[self.pelvis_id-1,2])

        return reward
    
    def _angular_velocity_penalty(self,
                                  body_vel:Motion) -> jax.Array:
        """
        penalty term for x-y plane angular velocity

        increase stability and reduce spinning

        formula:
            q_ang = -angular_velocity^2
        """
        # calculate reward
        reward = jp.sum(jp.square(body_vel.ang[self.pelvis_id-1,:2]))

        return reward
    
    def _joint_torque_penalty(self,
                              joint_torque:jax.Array) -> jax.Array:
        """
        penalty term for L2 norm of the total torques

        L2 norm: square root of the sum of the squares of the elements
        it provides a single scalar reprsents the magnitude of the torque vector
        in this case it means the strength or energy of the torques applied by all actuators combined
        this can encourage the robot to use less overall torque to achieve its objectives
        therefore discourage from applying large torques across multiple actuators

        """
        # calculate L2 norm of torque
        L2_norm = jp.sqrt(jp.sum(jp.square(joint_torque)))
        
        # calculate reward
        reward = L2_norm

        return reward
    
    def _action_rate_penalty(self,
                             action:jax.Array,
                             last_action:jax.Array) -> jax.Array:
        """
        penalty term for the rate of change of the action

        discourage agressive action and encourage smooth control

        formula:
            q_rate = -(action-last_action)^2
        """
        # calculate reward
        reward = jp.mean((action - last_action)**2)

        return reward
    
    def _motion_standing_still_penalty(self,
                               command:jax.Array,
                               joint_angle:jax.Array) -> jax.Array:
        """
        penalty term for offset of robot when no command

        encourage robot to stand still when no command

        formula:
            q_still = -offset^2
        """
        # calculate the error
        # the second term represent when small or no command are given
        reward = jp.mean((joint_angle - self.default_jnt_angle)**2) * (math.normalize(command[:2])[1] < 0.1)

        return reward

    def _alive_reward(self,
                      done:jax.Array) -> jax.Array:
        """
        reward term for rlive

        encourage the robot to be alive longer
        """
        return 1-done
    
    def _feet_air_time_reward(self,
                              air_time:jax.Array,
                              first_contact:jax.Array,
                              command:jax.Array) -> jax.Array:
        """
        reward term for spending time in the air, encouraging taking steps

        formula:
            q_air = sum(t_air - 0.5)
        """

        # obtain the reward when it making the first contact
        reward = jp.sum((air_time - 0.5) * first_contact)

        # check if there is a command
        reward *= (math.normalize(command[:2])[1] > 0.1)

        return reward
    
    def _single_leg_on_ground_reward(self,
                                     contact:jax.Array,
                                     command:jax.Array) -> jax.Array:
        """
        reward term for encouraging the agent to keep one foot in contact with ground at all time
        """
        # check if single_contact
        singe_contact = jp.sum(contact) == 1

        # update reward
        reward = 1.0*singe_contact*(math.normalize(command[:2])[1] > 0.1)

        return reward
    
    def _early_termination_penalty(self,
                                   done:jax.Array,
                                   step:jax.Array) -> jax.Array:
        """
        penalty term for early termination

        discourage the agent from terminating the episode early
        """
        # calculate reward
        terminal_early = done * (step < 950)
        reward = (950 - step) * terminal_early

        return reward
    
    def _both_foot_air_penalty(self,
                               contact:jax.Array) -> jax.Array:
        """
        penalty term for both foot in the air
        """
        both_air = jp.sum(contact) == 0
        return both_air 
    
    def _joint_limit_penalty(self,
                            limited_angle:jax.Array) -> jax.Array:
        """
        penalty for too close to joint limit
        """
        # set the penalty limit to 95% of the joint limit
        limit = self.joint_limit * 0.95

        # calculate the joint angles has larger or smaller than the limit
        out_of_limit = -jp.clip(limited_angle-limit[:,0], a_max=0.,a_min=None)
        out_of_limit += jp.clip(limited_angle-limit[:,1], a_max=None,a_min=0.)

        # calculate the reward
        reward = jp.sum(out_of_limit)

        return reward
    
    def _orientation_penalty(self,
                             body_pos:Transform) -> jax.Array:
        """
        penalty term for orientation of the robot
        """
        # define global up direction
        up = jp.array([0.0,0.0,1.0])

        # calculate the local up direction
        rot_up = math.rotate(up,body_pos.rot[self.pelvis_id-1])

        # calculate the error
        reward = jp.sum(jp.square(rot_up[:2]))

        return reward
    
    def _feet_position_penalty(self,
                               body_pos:Transform) -> jax.Array:
        """
        penalty term for feet to cross over
        """
        # define global y-axis
        global_y = jp.array([0.0,1.0,0.0])

        # calculate the local y-axis
        local_y = math.rotate(global_y,math.quat_inv(body_pos.rot[self.pelvis_id-1]))

        # calculate the local feet position relatively to pelvis
        left_feet_pos = body_pos.pos[self.left_foot_id-1] - body_pos.pos[self.pelvis_id-1]
        right_feet_pos = body_pos.pos[self.right_foot_id-1] - body_pos.pos[self.pelvis_id-1]

        # ignore z-axis
        local_y = local_y[:2]
        left_feet_pos = left_feet_pos[:2]
        right_feet_pos = right_feet_pos[:2]

        # project the feet position to local y-axis
        left_feet_y= jp.dot(left_feet_pos,local_y)/math.normalize(local_y)[0]
        right_feet_y= jp.dot(right_feet_pos,local_y)/math.normalize(local_y)[0]

        # check its local position
        # left feet should have a positive value, right feet should have a negative value
        reward = left_feet_y[1] < 0
        reward |= right_feet_y[1] > 0

        return reward
    
    def _z_position_penalty(self,
                            body_pos:Transform) -> jax.Array:
        """
        penalty term for too low or high positon of pelvis
        """
        # calculate the z position error
        reward = jp.abs(1-body_pos.pos[self.pelvis_id-1,2])

        return reward
    
    def _orientation_walking_penalty(self,
                                     orientation:jax.Array,
                                     body_pos:Transform) -> jax.Array:
        """
        penalty term for orientation of the robot
        """
        # define global y-axis
        global_x = jp.array([1.0,0.0,0.0])

        # calculate the local y-axis
        local_x = math.rotate(global_x,body_pos.rot[self.pelvis_id-1])

        # ignore z-axis
        local_x = local_x[:2]

        # calculate the error
        reward = jp.sum(jp.abs(orientation - local_x))

        return reward
        
envs.register_environment('Cassie', CassieEnv)
env_name = 'Cassie'
env = envs.get_environment(env_name)
eval_env = envs.get_environment(env_name)

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(128, 128, 128, 128))
pre_model_path = '/home/icepomelo/Code/policy_walking_final'
pre_model = model.load_params(pre_model_path)
# previous_params=pre_model
train_fn = functools.partial(
      ppo.train, num_timesteps=200000000,num_evals=10,
      reward_scaling=1, episode_length=1000, normalize_observations=True,
      action_repeat=1, unroll_length=20, num_minibatches=64,
      num_updates_per_batch=4, discounting=0.99, learning_rate=3.0e-4,
      entropy_cost=1e-2, num_envs=2048, batch_size=512,
      network_factory=make_networks_factory,previous_params=pre_model)

x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics['eval/episode_reward'])
    plt.xlim([0, train_fn.keywords['num_timesteps']])
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.title(f'y={y_data[-1]:.3f}')
    plt.plot(x_data, y_data)
    plt.show()

make_inference_fn, params, _= train_fn(environment=env,
                                       progress_fn=progress,
                                       eval_env=eval_env)

# save params
model_path = '/home/icepomelo/Code/policy_walking_final1'
model.save_params(model_path, params)
full_path = '/home/icepomelo/Code/inference_walking_final1'
# save inference func
with open(full_path, 'wb') as f:
    dill.dump(make_inference_fn, f)

print(f'time to jit: {times[1] - times[0]}')
print(f'time to train: {times[-1] - times[1]}')
#%%