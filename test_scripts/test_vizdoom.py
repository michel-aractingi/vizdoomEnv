#!/usr/bin/env python

from vizdoom import *
import random
import time
import argparse
import os 
import cv2
from utils import detect_collision
import numpy as np

joystick_actions = [vizdoom.Button.MOVE_FORWARD, vizdoom.Button.MOVE_BACKWARD,
                    vizdoom.Button.TURN_LEFT, vizdoom.Button.TURN_RIGHT, vizdoom.Button.ATTACK]

ammos = [GameVariable.AMMO0, GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3, GameVariable.AMMO4, GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7, GameVariable.AMMO8, GameVariable.AMMO9] 

def str2bool(x):  return x.lower() in ('true','1','t','yes','y')

def render_buffers(state):

    depth = state.depth_buffer
    if depth is not None:
        cv2.imshow('ViZDoom Depth Buffer', depth)

    # Labels buffer, always in 8-bit gray channel format.
    # Shows only visible game objects (enemies, pickups, exploding barrels etc.), each with unique label.
    # Labels data are available in state.labels, also see labels.py example.
    labels = state.labels_buffer
    if labels is not None:
        cv2.imshow('ViZDoom Labels Buffer', labels)

    # Map buffer, in the same format as screen buffer.
    # Shows top down map of the current episode/level.
    automap = state.automap_buffer
    if automap is not None:
        cv2.imshow('ViZDoom Map Buffer', np.rollaxis(automap,0,3))

    cv2.waitKey(28)

def get_distance_to_goals(game):

    i = list(range(5))
    state = game.get_state()
    actor_pos = np.array([game.get_game_variable(GameVariable.POSITION_X), game.get_game_variable(GameVariable.POSITION_Y)])
    goal_pos = np.array([[doom_fixed_to_float(state.game_variables[2 * it + 2]), \
                          doom_fixed_to_float(state.game_variables[2 * it + 3])] for it in i])
    N = doom_fixed_to_float(state.game_variables[-1])
    l2_dists = [1 - np.linalg.norm(actor_pos - goal_pos[it]) / N for it in i]

    return l2_dists

def random_env(game):

    game.init()
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

    game.set_window_visible(True)
    game.set_mode(vizdoom.Mode.SPECTATOR)
    game.set_screen_resolution(vizdoom.ScreenResolution.RES_1280X800)

    game.init()
    episodes = 10

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        game.new_episode()
        old_pos = np.array([game.get_game_variable(GameVariable.POSITION_X),
                            game.get_game_variable(GameVariable.POSITION_Y)])
        while not game.is_episode_finished():
            state = game.get_state()
            render_buffers(state)
            game.advance_action()
            last_action = game.get_last_action()
            reward = game.get_last_reward()
            if detect_collision(game):
                print("Collision Occuring...........!!!!!!")
            print("State #" + str(state.number))
            if not game.is_episode_finished():
                print('goal_cond reward: ', get_distance_to_goals(game))
                actor_pos = np.array([game.get_game_variable(GameVariable.POSITION_X),
                                      game.get_game_variable(GameVariable.POSITION_Y)])
                print("Actor Pos: ", actor_pos)
                print('Actor dist:', np.linalg.norm(actor_pos - old_pos))
                old_pos = np.array([game.get_game_variable(GameVariable.POSITION_X),
                                    game.get_game_variable(GameVariable.POSITION_Y)])
                Ammo = [game.get_game_variable(a) for a in ammos]
                print("Game variables ammo: ", Ammo)
            print("Action:", last_action)
            print("Reward:", reward)
            print("Done Flag:", game.is_episode_finished())
            print("=====================")
            if state.number == 500000:
                import pudb; pudb.set_trace()
        print("Episode finished!")
        print("Total reward:", game.get_total_reward())
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
    game.load_config(os.path.join(args.configdir, args.config))
    game.set_doom_scenario_path(os.path.join(args.configdir, args.wad))
    game.set_episode_timeout(4200)
    game.set_automap_buffer_enabled(True)
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
    parser.add_argument('--configdir', type=str, default="/home/maractin/Workspace/vizdoomEnv/scenarios/",
                        help='Specify config file path')
    parser.add_argument('--config', '-cfg', type=str, default="basic.cfg",
                        help='Specify config file')
    parser.add_argument('--wad', '-w', type=str, default="basic.wad",
                        help='Specify Wad file')
    parser.add_argument('--mode', type=str, default='random',
                        help='Specify whether to run interactively through keyboard or joystick or with random actions')

    args = parser.parse_args()

    main(args)
