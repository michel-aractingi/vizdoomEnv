import sys
from doomUtils import default_config
from mazeEnv import MazeEnv
from shootEnv import ShootEnv

def main(args):

    flags = default_config().parse_args(args)
    env = ShootEnv(flags)
    import pudb; pudb.set_trace()

if __name__=='__main__':
    main(sys.argv[1:])
