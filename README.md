# vizdoomEnv
Contains Gym wrapper for [VizDoom](https://github.com/mwydmuch/ViZDoom) with some test scripts.

To test vizdoomEnv:
```
python main.py --render=True
```
Make sure to have [pudb](https://pypi.org/project/pudb/) which is a python debugger. A breakpoint is set after 
initializing the environment and a ipython notebook will open so you can observe the variables and control the environment
through the *env* variable. 

To test controlling vizdoom interactively run:
```
python ./test_scripts/test_vizdoom.py --mode k --wad longhall.wad 
python ./test_scripts/test_vizdoom.py --mode j --wad longhall.wad
```
Mode  (k)  for keyboard. Mode  (j)   for joystick. Mode  (r)  for random actions.

By specifiying the *wad* file you can choose which map to use from [scenarios](https://es.naverlabs.com/michel-aractingi/vizdoomEnv/tree/master/scenarios).
