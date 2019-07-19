import numpy as np

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

