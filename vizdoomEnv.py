#Create a gym style environment around vizdoom

import numpy as np
from copy import deepcopy
import collections
import os
import random

from gym import core
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_binary import MultiBinary

from vizdoom import DoomGame
from vizdoom import GameVariable, doom_fixed_to_float
from utility import collision_detected, rgb2gray
import cv2

Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])
HEIGHT, WIDTH =120, 160
OFFSET = np.array([-192.0, -160.0])

class vizdoomEnv(core.Env):
    def __init__(self,args, process=None):
        '''
        TODO init vizdoom with config/ figure how to represent the config file
        '''
        self._config = deepcopy(args)

        #initialize DoomGame
        self.game = DoomGame()
        self.game.load_config(os.path.join(args.config_dir,args.config_file))
        if args.wad is not None: 
            self.game.set_doom_scenario_path(os.path.join(args.config_dir,args.wad))

        self._goal = None
        self._render = args.render
        self._init_seed = 0            # will be used if args.fix_seed is invoked#

        self.game.set_window_visible(self._render)
        self.game.set_depth_buffer_enabled(True)#args.use_depth)
        self.game.set_labels_buffer_enabled(True)#args.use_labels)
        self.game.set_automap_buffer_enabled(args.use_automap)

        try:
            if args.num_steps is not None:
                self.game.set_episode_timeout(args.num_steps)# + 1) #because state buffers will become None once reached the final observation
                self.game.set_episode_start_time(0)
        except AttributeError:
            pass

        self.game.add_game_args("+vid_forcesurface 1") #For a bug that might appear in multithreaded programs

        if process is not None:
            try:
                os.mkdir('process_{}'.format(process))
                os.chdir('./process_{}'.format(process))
            except FileExistsError:
                os.chdir('./process_{}'.format(process))
                try:
                    os.rmdir('_vizdoom'); os.remove('_vizdoom.ini')
                except FileNotFoundError:
                    pass
            self.init_seed = process * 2**10
            self.game.init()
            os.chdir('..')
        else:
            self.game.init()


        #initialize gym interface
        if self._config.flattened_obs:
            self._observation_space = Box(-np.inf,np.inf,(self._get_observation().shape[0],), dtype=np.float32)   #Box(low,high,shape=)
        else:
            self._observation_space = Box(-np.inf,np.inf,self._get_observation().shape, dtype=np.float32)   #Box(low,high,shape=)

        # add 1 for the noop action, this does not consider combination of actions

        if self._config.action_space.lower() in ('b', 'binary'):
            #Use MultiBinary action space for the robot to be able to use multiple discrete actions together. No need for +1 since its a vector
            self._action_space = MultiBinary(len(self.game.get_available_buttons()))
        elif self._config.action_space.lower() in ('d', 'discrete'):
            if len(self.game.get_available_buttons()) == 4:
                self._action_space = Discrete(9)
                from utility import discrete_action_dict
                self._action_dict = discrete_action_dict
            if len(self.game.get_available_buttons()) == 3:
                self._action_space = Discrete(4)#6)
                from utility import simple_discrete_action_dict
                self._action_dict = simple_discrete_action_dict
            else:
                raise NotImplementedError('Discrete action space with more than 4 actions not implemented yet')

        #self._action_space = Box(0,1,(self.game.get_available_buttons().shape[0],))  #continous action space not available now

        self._env_timestep = 0
        self._horizon = self.game.get_episode_timeout() - self.game.get_episode_start_time()

        self._reset = False #should be set in reset method
        self._actor_pos = np.array([0.0, 0.0])
        self._ammo = 0.0 
        self._health = 0.0

        #Initialize dictionaries for info that will be filled in the step function
        self._info = {}
        self._info['episode'] = {}#collections.defaultdict(int)
        self._info['episode']['reward'] = 0
        self._info['episode']['collision'] = 0
        self._info['episode']['length'] = 0
        self._info['episode']['actor_pos'] = (0, 0)
        self._info['episode']['goal_pos'] = (0, 0)

    def reset(self):
        """
        TODO reset all env and prepare to start a new episode
    
            Return: observation
        """

        if self._config.fix_seed:
            self._init_seed = (self._init_seed + 1) % 2**32     # set_seed requires int
            self.game.set_seed(self._init_seed)

        self.game.new_episode()
        self._env_timestep = self.game.get_episode_time() - self.game.get_episode_start_time()
        self._reset = True
        self._info['episode'] = {}#collections.defaultdict(int)
        self._info['episode']['reward'] = 0
        self._info['episode']['collision'] = 0
        self._info['episode']['length'] = 0
        self._info['episode']['actor_pos'] = self._get_actor_pos()
        self._info['episode']['goal_pos'] = self._get_goal_pos()

        self._actor_pos = self._get_actor_pos(True)
        self._ammo = self.game.get_game_variable(GameVariable.AMMO2)
        self._health = self.game.get_game_variable(GameVariable.HEALTH)

        if self._config.task == 'goal_cond':
            self._goal_list = list(range(self._config.nb_goals))
            self._goal = random.choice(self._goal_list)
            self._info['episode']['goal'] = self._goal
            self._info['episode']['goals_done'] = 0

        return self._get_observation()

    def step(self, action):
        '''
        Perform a step with the given action
        Args:
            - action: integer between 0 and number of possible actions

        Return: observation, reward, info, done (just like gym)
        '''
        #print('vizdoom step',self.game.get_episode_time())
        if not self._reset:
            raise RuntimeError("env.reset was not called before running the environment")

        #decode action
        if isinstance(self.action_space,Discrete):
            if action not in range(self.action_space.n):
                raise ValueError("action should have value between (0 , {})".format(self.action_space.n))
            decoded_action = self._action_dict[action]

        elif isinstance(self.action_space,MultiBinary):
            try:
                if len(action) != self.action_space.shape[0]:
                    raise ValueError("MultiBinary actions should be a vector of length {}".format(self.action_space.shape[0]))
            except TypeError:
                raise TypeError('MultiBinary action space expects a list')
            decoded_action = list(action)

        #perform action
        game_reward = self.game.make_action(decoded_action)
        self._env_timestep += 1
        done = self.game.is_episode_finished() #or (self._env_timestep == self.game.get_episode_timeout() - 1)

        if self._config.task=='goal_cond' and self._goal not in self._goal_list:
            #if a goal is reached; don't stop the episode, sample a new goal
            if self._goal_list == []:
                done = True
            else:
                self._goal = random.choice(self._goal_list)
                self._info['episode']['goals_done'] += 1
                self._info['episode']['goal'] = self._goal

        if not done:
            observation = self._get_observation()
            reward = self._get_reward(game_reward)
        else:
            self._reset = False
            observation = None
            reward = 0

        self._info['episode']['reward'] += reward
        self._info['episode']['length'] += 1
        self._info['episode']['actor_pos'] = self._get_actor_pos(True)
        if not done and collision_detected(self.game.get_state().depth_buffer, forward_action=decoded_action[0]):
            self._info['episode']['collision'] += 1
            reward -= 0.1

        self._actor_pos = self._get_actor_pos()

        return Step(observation=observation, reward=reward, done=done, info=deepcopy(self._info))



    def render(self,mode='rgb_array'):
        ''' render different images (rgb, depth, automap, labelled objects) according to the config file
            MIGHT NOT WORK YET! fix...
        '''
        state = self.game.get_state()
        # This buffer is always available.
        # If we don't want to use RGB input we should just ignore it(can't disable it)
        if self._config.use_rgb:
            cv2.imshow('ViZDoom Screen Buffer', state.screen_buffer)

        # Depth buffer, always in 8-bit gray channel format.
        if self._config.use_depth is not None:
            cv2.imshow('ViZDoom Depth Buffer', state.depth_buffer)

        # Labels buffer, always in 8-bit gray channel format.
        # Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
        if self._config.use_labels is not None:
            cv2.imshow('ViZDoom Labels Buffer', state.labels_buffer)

        # Map buffer, in the same format as screen buffer.
        # Shows top down map of the current episode/level.
        if self._config.use_automap is not None:
            cv2.imshow('ViZDoom Map Buffer', state.automap_buffer)

        cv2.waitKey(0)

        return state.screen_buffer

    def close(self):
        cv2.destroyAllWindows()

    def seed(self, seed):

        random.seed(seed)
        np.random.seed(seed)
        
        return seed

    def _get_reward(self, game_reward=0):
        '''
        TODO get reward of current timestep (function facilitate reward shaping)
        '''

        if self._config.task is None:
            reward = 2*game_reward/self.game.get_episode_timeout()

        elif self._config.task.lower() == 'goal_cond':
            state = self.game.get_state()
            N = doom_fixed_to_float(state.game_variables[-1])
            actor_pos = self._get_actor_pos()
            goal_pos = self._get_goal_pos()
            l2_dist = np.linalg.norm(actor_pos - goal_pos) / N

            if self._config.shaped_reward:
                #First living penalty:
                reward  = -0.008
                #Second dist penalty (not moving)/ dist reward for moving
                dist_actor = np.linalg.norm(actor_pos - self._actor_pos)
                if dist_actor < 0.5:
                    reward -= 0.03
                else:
                    reward += dist_actor * 9e-5
                reward += 0.01*(1 - l2_dist) # penalty on distance to goal
            else:
                reward = (1 - l2_dist)**self._config.exp - 0.3

            if self.game.get_game_variable(GameVariable.AMMO2) > self._ammo: #If ammo box picked up add reward or penalty
                if l2_dist < 0.1:
                    reward += 1
                    self._goal_list.remove(self._goal)
                else:
                    reward += -1
                    self._remove_wrong_goal()
                self._ammo = self.game.get_game_variable(GameVariable.AMMO2)

        elif self._config.task.lower() == 'maze':

            if self.game.get_game_variable(GameVariable.AMMO2) > self._ammo: #If ammo box picked up add reward or penalty
                reward = 1.0
                self._ammo = self.game.get_game_variable(GameVariable.AMMO2)
            else:
                reward = 0.0 

        else:
            raise NotImplementedError

        return round(reward, 6)

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

        if self._config.use_automap:
            automap = np.expand_dims(state.automap_buffer, axis=0)
            if obs is not None:
                obs = np.concatenate((obs, automap))
            else:
                obs = automap

        assert obs is not None, \
            'No visual input presented! Make sure to use at least one of the input maps (rgb, grayscale, depth, labels, automap)'

        if self._config.img_size is not None:
            im_size = self._config.img_size
            obs = np.rollaxis(obs,0,3)
            obs = cv2.resize(obs,(im_size,im_size))
            obs = np.rollaxis(obs,2,0)

        if self._config.flattened_obs:
            obs = obs.flatten()
            if self._config.num_state_vars > 0:
                #obs = np.concatenate((obs, state.game_variables))
                if len(state.labels) == 0:
                    obs = np.concatenate((obs, np.array([0.0]*4)))
                else:
                    L = state.labels[0]
                    S = np.array([L.x / WIDTH, L.y / HEIGHT, L.height, L.width])
                    obs = np.concatenate((obs, S))
            if self._config.task == 'goal_cond':
                obs = np.concatenate((obs, [self._goal]))

        return obs

    def _get_actor_pos(self, offset=True):
        '''

        return: numpy 2D array with x,y position of agent
        '''
        return np.array([self.game.get_game_variable(GameVariable.POSITION_X),
                         self.game.get_game_variable(GameVariable.POSITION_Y)]) - (OFFSET * int(offset))

    def _get_goal_pos(self, goal=None, offset=True):
        '''

        return: numpy 2D array with x,y position of the current goal
        '''
        #assert self._config.task == 'goal_cond', \
        #    'get_goal_pos works only for goal conditioned tasks'
        if self._config.task == 'goal_cond':
            goal_idx = self._goal if goal is None else goal
        else:
            goal_idx = 0
        state = self.game.get_state()
        goal_x = doom_fixed_to_float(state.game_variables[2 * goal_idx + 2])
        goal_y = doom_fixed_to_float(state.game_variables[2 * goal_idx + 3])

        return np.array([goal_x, goal_y]) - (OFFSET * int(offset))

    def _remove_wrong_goal(self):
        '''
        Removes one goal object when it is mistakenly picked up by the agent.
        In order to avoid problems while training in case this goal is assigned at a later stage.

        return: None
        '''
        actor_pos = self._get_actor_pos()
        goal_poses = [self._get_goal_pos(i) for i in self._goal_list]
        dists = [np.linalg.norm(actor_pos - g_pos) for g_pos in goal_poses]
        closest_goal = self._goal_list[np.argmin(dists)]
        self._goal_list.remove(closest_goal)

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
