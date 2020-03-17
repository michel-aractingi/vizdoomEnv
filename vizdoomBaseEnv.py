#Create a gym style environment around vizdoom

import numpy as np
from copy import deepcopy
import os
import random

from gym import core
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_binary import MultiBinary

from vizdoom import DoomGame
from utility import collision_detected, rgb2gray

CONFIG_DIR='/home/maractin/Workspace/vizdoomEnv/scenarios/'

class vizdoomBaseEnv(core.Env):
    def __init__(self, args, game=None):
        self._config = deepcopy(args)

        #initialize DoomGame
        self.game = DoomGame() if game is None else game 
        self.game.load_config(os.path.join(CONFIG_DIR,args.config))
        self.game.set_doom_scenario_path(os.path.join(CONFIG_DIR,args.wad))
        self.game.set_episode_timeout(args.episode_length)

        self.game.set_window_visible(args.render)
        self.game.set_depth_buffer_enabled(args.render_depth)
        self.game.set_labels_buffer_enabled(args.render_labels)

        self.game.init(); os.system('rm -rf _vizdoom*')


        self._observation_space = Box(low=-np.inf,high=np.inf,
                shape=self._get_observation().shape, dtype=np.float32)
        
        #TODO add continuous actions
        if self._config.action_space.lower() in ('b', 'binary'):
            self._action_space = MultiBinary(args.num_actions)
        elif self._config.action_space.lower() in ('d', 'discrete'):
            self._action_space = Discrete(args.num_actions)
        else:
            raise NotImplementedError('Invalid action space')
        
        self._env_timestep = 0
        self._horizon = self.game.get_episode_timeout() - \
                        self.game.get_episode_start_time()

        self._render = args.render
        self._reset = False 

        self._info = EpisodicDict()

    def reset(self):
        """
        reset all env and prepare to start a new episode
            Return: observation
        """

        self.game.new_episode()
        self._env_timestep = self.game.get_episode_time() - self.game.get_episode_start_time()
        self._reset = True
        self._info.reset()

        return self._get_observation()


    def step(self, action):
        pass

    def render(self,mode='rgb_array'):
        """ 
        render different images (rgb, depth, labelled objects) 
         according to the config file
        """

        state = self.game.get_state()
        # This buffer is always available.
        if self._config.use_rgb:
            cv2.imshow('ViZDoom Screen Buffer', state.screen_buffer)

        # Depth buffer, always in 8-bit gray channel format.
        if self._config.use_depth is not None:
            cv2.imshow('ViZDoom Depth Buffer', state.depth_buffer)

        # Labels buffer, always in 8-bit gray channel format.
        if self._config.use_labels is not None:
            cv2.imshow('ViZDoom Labels Buffer', state.labels_buffer)

        cv2.waitKey(0)

        return state.screen_buffer

    def close(self):
        cv2.destroyAllWindows()
        super(vizdoomBaseEnv, self).close()

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def _step_discrete(self, action):

        assert action in range(self.action_space.n), ValueError(
                "Actions should have value between (0 , {})".format(self.action_space.n))
        action_oneHot = [0]*self.action_space.n
        action_oneHot[action] = 1

        return self.game.make_action(action_oneHot)

    def _step_multibinary(self, action):

        assert len(action) == self.action_space.shape[0], ValueError(
                'Actions of len {} is invalid for {}-dim space'.format(
                    len(action),self.action_space.shape[0]))

        return self.game.make_action(list(action))

    def _get_observation(self):
        '''
        get observation (RGB + Depth + game variables) depending on config

            Return: 1D flattened Array with size of the _observation_space
        '''

        state = self.game.get_state()
        obs = None
        if self._config.use_rgb:
            obs = state.screen_buffer
        if self._config.use_grayscale:
            obs = np.expand_dims(rgb2gray(state.screen_buffer), axis=0)

        if self._config.use_depth:
            depth = np.expand_dims(state.depth_buffer, axis=0)
            if obs is not None:
                obs = np.concatenate((obs, depth))
            else:
                obs = depth

        if self._config.use_labels:
            labels = np.expand_dims(state.labels_buffer, axis=0)
            if obs is not None:
                obs = np.concatenate((obs, labels))
            else:
                obs = labels

        assert obs is not None, \
            'No visual input presented! Make sure to use at least one of the input maps (rgb, grayscale, depth, labels)'
    
        return obs 
   
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def id(self):
        return self._id

    @property
    def horizon(self):
        return self._horizon


class EpisodicDict(dict):

    def __init__(self, **kwargs):
        super(EpisodicDict, self).__init__()

        self['reward'] = 0
        self['collision'] = 0
        self['length'] = 0
        for key, value in kwargs.items():
            self[key] = value

    def update(self, rewards, collisions, length=1, **kwargs):

        self['reward'] += rewards
        self['collision'] += collisions
        self['length'] += length
        for key, value in kwargs.items():
            self[key] = value

    def reset(self, **kwargs):

        self['reward'] = 0
        self['collision'] = 0
        self['length'] = 0
        for key, value in kwargs.items():
            self[key] = value


