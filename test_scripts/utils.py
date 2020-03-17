import numpy as np
from vizdoom import * 
import cv2 

joystick_actions = [vizdoom.Button.MOVE_FORWARD, vizdoom.Button.MOVE_BACKWARD,
                    vizdoom.Button.TURN_LEFT, vizdoom.Button.TURN_RIGHT, vizdoom.Button.ATTACK]

ammos = [GameVariable.AMMO0, GameVariable.AMMO1, GameVariable.AMMO2, GameVariable.AMMO3, GameVariable.AMMO4, GameVariable.AMMO5, GameVariable.AMMO6, GameVariable.AMMO7, GameVariable.AMMO8, GameVariable.AMMO9]

def str2bool(x):  return x.lower() in ('true','1','t','yes','y')

def compress_depthmap(d):
    return np.max(d,axis=0)

def is_last_action_forward(game):
    
    buttons = [str(b) for b in game.get_available_buttons()]
    fwd_idx = buttons.index('Button.MOVE_FORWARD')
    return bool(game.get_last_action()[fwd_idx])

def detect_collision(game):
    
    #First strategy compression
    print(game,game.get_state())
    state = game.get_state()
    if state is None:     # Required fix for when the episode is finished the get_state returns None
        return False
    depth = state.depth_buffer
    compressed_depth = compress_depthmap(depth)
    len_compressed = compressed_depth.shape[0]
    crop_ratio = len_compressed//4
    if np.min(compressed_depth[crop_ratio:len_compressed - crop_ratio]) == 0 and is_last_action_forward(game):
        return True
    return False

def plot_sectors(state):
    import matplotlib.pyplot as plt
    for s in state.sectors:
        print("Sector floor height:", s.floor_height, "ceiling height:", s.ceiling_height)
        print("Sector lines:", [(l.x1, l.y1, l.x2, l.y2, l.is_blocking) for l in s.lines])

        # Plot sector on map
        for l in s.lines:
            if l.is_blocking:
                plt.plot([l.x1, l.x2], [l.y1, l.y2], color='black', linewidth=2)

    # Show map
    plt.show()

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


def color_labels(labels):
    """
    Walls are blue, floor/ceiling are red (OpenCV uses BGR).
    """
    tmp = np.stack([labels] * 3, -1)
    tmp[labels == 0] = [255, 0, 0]
    tmp[labels == 1] = [0, 0, 255]
    return tmp

