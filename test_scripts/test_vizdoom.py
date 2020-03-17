#!/usr/bin/env python

from vizdoom import *
import random
import time
import argparse
import os 
import cv2
from utils import *
import numpy as np

CONFIG_DIR = '/home/maractin/Workspace/vizdoomEnv/scenarios/'

def random_env(game):

    game.init(); os.system('rm -rf _vizdoom*')
    import pudb; pudb.set_trace()
    shoot = [0, 0, 1]
    left = [1, 0, 0]
    right = [0, 1, 0]

    actions = [shoot, left, right]

    episodes = 10
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            depth = state.depth_buffer
            misc = state.game_variables
            reward = game.make_action(random.choice(actions))
            print ("\treward:", reward)
            time.sleep(0.02)
        print ("Result:", game.get_total_reward())
        time.sleep(2)

def keyboard_env(game):

    game.set_mode(vizdoom.Mode.SPECTATOR)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_1280X800)
    game.init()

    episodes = 10
   
    for i in range(episodes):

        print("Episode #" + str(i + 1))
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()
            print("State #{}, Action {}, Reward {}, Done {}".format(
                str(state.number),last_action,reward,game.is_episode_finished()))
            print("=====================")
        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
        print("Total kills:", game.get_game_variable(GameVariable.KILLCOUNT))
        print("************************")
        time.sleep(2.0)

    game.close()

def joystick_env(game):
    #Modify the action space accordingly
    game.clear_available_buttons()
    [game.add_available_button(B) for B in joystick_actions]
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_1280X800)
    game.init()

    from joystick_handler import get_triggered_button
    episodes = 10
    for i in range(episodes):
        game.new_episode()
        action = [0, 0, 0, 0, 0]
        while not game.is_episode_finished():

            state = game.get_state()
            render_buffers(state)

            action = get_triggered_button(action)  # get_action(last_action)
            reward = game.make_action(action)

            print('action:', action)
            print("\treward:", reward)
            time.sleep(0.02)
        print("Result:", game.get_total_reward())
        time.sleep(2)


def main(args):

    game = DoomGame()
    game.set_window_visible(True)
    game.load_config(os.path.join(CONFIG_DIR, args.config))
    game.set_doom_scenario_path(os.path.join(CONFIG_DIR, args.wad))
    game.set_episode_timeout(4200)
    #game.init() don't initialize the game here in case each mode needs its own configuration

    if args.pudb:
        import pudb; pudb.set_trace()

    if args.mode.lower() in ('keyboard','k'):
        keyboard_env(game)
    elif args.mode.lower() in ('random', 'r'):
        random_env(game)
    elif args.mode.lower() in ('joystick', 'j'):
        joystick_env(game)
    else:
        raise ValueError('Invalid mode. Must be either keyboard or random or joystick.')

    game.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser([])
    parser.add_argument('-pudb', type=str2bool, default=False,
                        help='Whether to use pudb or not')
    parser.add_argument('--config', type=str, default="basic.cfg",
                        help='Specify config file')
    parser.add_argument('--wad', type=str, default="basic.wad",
                        help='Specify Wad file')
    parser.add_argument('--mode', type=str, default='random',
                        help='Specify whether to run interactively through keyboard or joystick or with random actions')

    args = parser.parse_args()
    
    main(args)
