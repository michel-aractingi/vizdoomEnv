#Create a gym style environment around vizdoom

import numpy as np
from copy import deepcopy
import os
import random

from gym.spaces import Box, Discrete, MultiBinary
from vizdoomBaseEnv import EpisodicDict, vizdoomBaseEnv
from vizdoom import GameVariable
from doomUtils import collision_detected, Step

class ShootEnv(vizdoomBaseEnv):
    def __init__(self, args):
        '''
        TODO init vizdoom with config/ figure how to represent the config file
        '''

        super(ShootEnv, self).__init__(args)
        self._init_seed = 0            # will be used if args.fix_seed is invoked#

        if self._config.flattened_obs:
            self._observation_space = Box(low=-np.inf, high=np.inf,
                    shape=(self._get_observation().shape[0],), dtype=np.float32)

        self._killcount = 0.0
        self._ammo = 0.0 
        self._health = 0.0
        
        self._xwidth = self.game.get_screen_width()

    def reset(self):
        """
        TODO reset all env and prepare to start a new episode
    
            Return: observation
        """
        
        if self._config.fix_seed:
            self._init_seed = (self._init_seed + 1) % 2**32     # set_seed requires int
            self.game.set_seed(self._init_seed)

        super(ShootEnv, self).reset()

        self._killcount = 0.0
        self._ammo = self.game.get_game_variable(GameVariable.AMMO2)
        self._health = self.game.get_game_variable(GameVariable.HEALTH)

        return self._get_observation()

    def step(self, action):
        '''
        Perform a step with the given action
        Args:
            - action: integer between 0 and number of possible actions

        Return: observation, reward, info, done (just like gym)
        '''

        if not self._reset:
            raise RuntimeError("env.reset was not called before running the environment")

        shootbool = self._shootbool()
        if isinstance(self.action_space, MultiBinary):
            game_reward = self._step_multibinary(action, shootbool)
        elif isinstance(self.action_space, Discrete):
            game_reward = self._step_discrete(action,shootbool)
        else:
            raise NotImplementedError

        self._env_timestep += 1
        episode_finished = self.game.is_episode_finished()

        if episode_finished:
            self._reset = False
            observation = None
            if self.game.is_player_dead():
                reward = -1.0
            else:
                reward =  0.0
             
            return Step(observation, reward, True, deepcopy(self._info))
            
        observation = self._get_observation()
        reward = self._get_reward(game_reward)
        
        collision = 0
        if self._config.use_depth and collision_detected(self.game):
            collision = 1
            reward -= 0.1
        self._info.update(reward, collision)

        return Step(observation, reward, False, deepcopy(self._info))


    def _get_reward(self, game_reward=0):
        '''
        TODO get reward of current timestep (function facilitate reward shaping)
        '''
        #reward = game_reward
        reward = 0.0
        if self.game.get_game_variable(GameVariable.KILLCOUNT) > self._killcount:
            self._killcount += 1
            reward += 0.1

        return reward

    def _get_observation(self):

        obs = super(ShootEnv, self)._get_observation()
        #if self._config.img_size is not None:
        #    im_size = self._config.img_size
        #    obs = np.rollaxis(obs,0,3)
        #    obs = cv2.resize(obs,(im_size,im_size))
        #    obs = np.rollaxis(obs,2,0)

        if self._config.flattened_obs:
            obs = obs.flatten()
            if self._config.num_state_vars > 0:
                obs = np.concatenate((obs, np.array([0.0]*4)))

        return obs

    def _shootbool(self):
        
        state = self.game.get_state()
        for obj in state.labels:
            if obj.object_name == 'RemovableDoomImp':
                if obj.x <= self._xwidth/2 <= obj.x + obj.width:
                    return True
        return False

    def _step_discrete(self, action, shootbool):

        assert action in range(self.action_space.n), ValueError(
                "Actions should have value between (0 , {})".format(self.action_space.n))
        action_oneHot = [0]*(self.action_space.n + 1) # augment shoot action
        action_oneHot[int(action)] = 1
        action_oneHot[-1] = int(shootbool)

        return self.game.make_action(action_oneHot)

    def _step_multibinary(self, action, shootbool):
        assert len(action) == self.action_space.shape[0], ValueError(
                'Actions of len {} is invalid for {}-dim space'.format(
                    len(action),self.action_space.shape[0]))
        action = list(action)
        action.append(int(shootbool))

        return self.game.make_action(action)

