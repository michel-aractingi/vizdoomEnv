import os
import argparse
import random

parser = argparse.ArgumentParser([])
parser.add_argument('--name', type=str, default=None,
                    help='Name of the maze.')
parser.add_argument('--num_mazes', type=int, default=100,
                    help='Number of the mazes to generate.')
parser.add_argument('--textures_path', type=str, default="/home/maractin/Workspace/vizdoomEnv/Textures2.txt",
                    help='Specify config file path')
parser.add_argument('--save_dir', type=str, default="/home/maractin/Workspace/vizdoomEnv/scenarios/R_maze/",
                    help='Specify config file path')

args = parser.parse_args()

with open(args.textures_path, 'r') as f:
    textures_list = f.readlines()
    f.close()

texturefloor   = 'texturefloor'
textureceiling = 'textureceiling'
texturemiddle  = 'texturemiddle'
Btexturefloor   = b'texturefloor'
Btextureceiling = b'textureceiling'
Btexturemiddle  = b'texturemiddle'
encoding = 'utf-8'

with open(os.path.join(args.save_dir, args.name), 'rb') as f:
    Map = f.readlines()
    f.close()

for j in range(args.num_mazes):
    for i in range(len(Map)):
        '''
        if Btexturefloor in Map[i]:
            text = random.choice(textures_list)[:-1]
            new = texturefloor + '="{}"'.format(text) + ';\n'
            Map[i] = bytes(new, encoding)    
        elif Btextureceiling in Map[i]:
            text = random.choice(textures_list)[:-1]
            new = textureceiling + '="{}"'.format(text) + ';\n'
            Map[i] = bytes(new, encoding)
        '''
        if Btexturemiddle in Map[i]:
            text = random.choice(textures_list)[:-1]
            new = texturemiddle + '="{}"'.format(text) + ';\n'
            Map[i] = bytes(new, encoding)
        
    with open(os.path.join(args.save_dir, args.name[:-4] + '_{}.wad'.format(j)), 'wb') as f:
        for l in Map:
            f.write(l)
        f.close()
