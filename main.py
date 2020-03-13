import sys
from utility import default_config
from mazeEnv import mazeEnv

def main(args):

    flags = default_config().parse_args(args)
    env = mazeEnv(flags)
    import pudb; pudb.set_trace()

if __name__=='__main__':
    main(sys.argv[1:])
