import argparse
import numpy as np
import collections

action_dict = { \
    0 : [0, 0, 0, 0, 0, 0], #Noop
    1 : [1, 0, 0, 0, 0, 0], #MOVE_FORWARD
    2 : [0, 1, 0, 0, 0, 0], #MOVE_BACKWARD
    3 : [0, 0, 1, 0, 0, 0], #MOVE_LEFT
    4 : [0, 0, 0, 1, 0, 0], #MOVE_RIGHT
    5 : [0, 0, 0, 0, 1, 0], #TURN_LEFT
    6 : [0, 0, 0, 0, 0, 1]  #TURN_RIGHT
    }
action_sym_dict = { \
    0 : 'Noop',
    1 : 'MOVE_FORWARD',
    2 : 'MOVE_BACKWARD',
    3 : 'MOVE_LEFT',
    4 : 'MOVE_RIGHT',
    5 : 'TURN_LEFT',
    6 : 'TURN_RIGHT'
    }
discrete_action_dict = { \
    0 : [0, 0, 0, 0], #Noop
    1 : [1, 0, 0, 0], #MOVE_FORWARD
    2 : [0, 1, 0, 0], #MOVE_BACKWARD
    3 : [0, 0, 1, 0], #TURN_LEFT
    4 : [0, 0, 0, 1], #TURN_RIGHT
    5 : [0, 1, 0, 1], #MB + TR
    6 : [0, 1, 1, 0], #MB + TL
    7 : [1, 0, 0, 1], #MF + TR
    8 : [1, 0, 1, 0]  #MF + TL
    }

simple_discrete_action_dict = { \
    0 : [0, 0, 0], #Noop
    1 : [1, 0, 0], #MOVE_FORWARD
    2 : [0, 1, 0], #TURN_LEFT
    3 : [0, 0, 1], #TURN_RIGHT
    4 : [1, 1, 0], #MF + TL
    5 : [1, 0, 1], #MF + TR
    }

vargs_list = ['config_dir','config_file','wad','use_rgb',
              'use_depth','use_labels','use_automap','render',
              'flattened_obs','img_size', 'num_state_vars',
              'task', 'nb_goals', 'action_space', 'exp',
              'shaped_reward', 'use_grayscale', 'fix_seed']


Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])

def str2bool(x):
    if x.lower() in ('true','t','y','yes','1'):
        return True
    elif x.lower() in ('false','f','n','no','0'):
        return False
    else:
        raise TypeError('Invalid boolean value')

def default_config(parser = argparse.ArgumentParser([])):

    #VizDoom args
    parser.add_argument('--config','-cfg', type=str, default="gymEnv.cfg",
                        help='Specify config file')
    parser.add_argument('--wad','-w', type=str, default=None,
                        help='Specify wad file (should be in config_dir)')
    parser.add_argument('--episode_length', type=int, default=50,
                        help='Horizon of the episode.')

    #State space args
    parser.add_argument('--use_rgb', action='store_true', default=True,
                        help='Specify whether to add rgb input to the state space')
    parser.add_argument('--use_grayscale', action='store_true', default=False,
                        help='Specify whether to use grayscale input to the state space')
    parser.add_argument('--render_depth', action='store_true', default=False,
                        help='render depth channel')
    parser.add_argument('--render_labels', action='store_true', default=False,
                        help='render labels information')
    parser.add_argument('--use_depth', action='store_true', default=False,
                        help='Specify whether to add depth map input to the state space')
    parser.add_argument('--use_labels', action='store_true', default=False,
                        help='Specify whether to add object labels map (segmented) input to the state space')
    parser.add_argument('--flattened_obs', '-fobs', type=str2bool, default=False,
                        help='Whether to present the observations as flattened array or not')
    parser.add_argument('--num_state_vars', type=int, default=0,
                        help='Whether to add state variables to the observation space')

    #Action space args
    parser.add_argument('--action_space', type=str, default='d',
                        help='Specify which type of action space to use (binary, discrete, continuous)')
    parser.add_argument('--num_actions', type=int, default=4)

    #Visualization args
    parser.add_argument('--render', '-r', type=str2bool, default=False,
                        help='To enable rendering')
    parser.add_argument('--img_size', type=int, default=None,
                        help='Reshape vizdoom obs to square image of size.')

    #Task specific args
    parser.add_argument('--shaped_reward', type=str2bool, default=True,
                        help='Whether to use the shaped sparse reward')
    parser.add_argument('--task', type=str, default=None,
                        help='Which task to solve (Maze, goal-cond...)')
    parser.add_argument('--nb_goals', '-n', type=int, default=1,
                        help='Number of existing goals in an environment')
    parser.add_argument('--exp', '-e', type=int, default=1,
                        help='Reward exponent')

    #Random seed 
    parser.add_argument('--fix_seed', type=str2bool, default=False,
                        help='Fix a given seed and therefore fix the events in the episode')

    return parser


def compress_depthmap(d):
    return np.max(d, axis=0)


def is_last_action_forward(game):
    return bool(game.get_last_action()[0]) #Forward action should always be the first action


def collision_detected(game):

    compressed_depth = compress_depthmap(game.get_state().depth_buffer)
    forward_action = game.get_last_action()[0]
    len_compressed = compressed_depth.shape[0]
    crop_ratio = len_compressed // 4
    if np.min(compressed_depth[crop_ratio:len_compressed - crop_ratio]) < 2 and bool(forward_action):
        return True
    return False


def rgb2gray(rgb):
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
