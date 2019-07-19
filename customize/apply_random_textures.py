import omg
import sys
import random
import os
import argparse

FLOOR_TEXTURE = ['ZZWOLF1','MFLR8_1', 'MFLR8_2', 'MFLR8_3', 'FLOOR0_5']
CEILING_TEXTURE = ['ZIMMER8', 'MFLR8_1', 'CEIL4_1', 'MFLR8_2', 'MFLR8_3']
TRAIN_TEXTURES_PATH = './textures/train_textures.txt'
TEST_TEXTURES_PATH = './textures/test_textures.txt'
RANDOM_SEED = random.randint(100, 10000)
SUFFIX = '_manymaps_test.wad'

def get_textures(textures_file):
  with open(textures_file) as f:
    textures = f.read().split()
  return textures

def copy_attributes(in_map, out_map):
  to_copy = ['BEHAVIOR']
  for t in to_copy:
    if t in in_map:
      out_map[t] = in_map[t]

def set_floor_ceiling(map_editor):
  for s in map_editor.sectors:
    s.tx_floor = random.choice(FLOOR_TEXTURE)
    s.tx_ceil = random.choice(CEILING_TEXTURE)

def set_walls(map_editor, textures):
  for s in map_editor.sidedefs:
    s.tx_mid = random.choice(textures)

def inner_change_textures(map_editor, textures):
  set_walls(map_editor, textures)
  set_floor_ceiling(map_editor)

def change_textures(in_map, textures):
  map_editor = omg.MapEditor(in_map)
  inner_change_textures(map_editor, textures)
  out_map = map_editor.to_lumps()
  #copy_attributes(in_map, out_map)
  return out_map

def add_map_to_wad(wad, map, index):
  wad.maps['MAP%.2d' % (index + 2)] = map

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', help='input wad file to process')
  parser.add_argument('--mode', choices=['train', 'test'], help='what those maps are going to be used for: train or test?')
  parser.add_argument('--num_maps', type=int, default=10, help='Number of the mazes to generate.')
  
  args = parser.parse_args()
  in_file = args.input
  out_file = in_file[:-4]
  wad = omg.WAD(in_file)
  
  if args.mode == 'test':
    textures = get_textures(TEST_TEXTURES_PATH)
  else:
    global FLOOR_TEXTURE, CEILING_TEXTURE
    FLOOR_TEXTURE, CEILING_TEXTURE = ['MFLR8_1'], ['MFLR8_1']
    textures = get_textures(TRAIN_TEXTURES_PATH)

  for index in range(args.num_maps):
    print(index)
    wad.maps['MAP01'] = change_textures(wad.maps['MAP01'], textures)
    wad.to_file(out_file + '_{}_{}.wad'.format(args.mode,index))

if __name__ == '__main__':
    main()
