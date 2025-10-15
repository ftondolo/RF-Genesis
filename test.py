import argparse
from termcolor import colored
import time
from genesis.raytracing import pathtracer
from genesis.raytracing import signal_generator

from genesis.environment_diffusion import environemnt_diff
from genesis.object_diffusion import object_diff
from genesis.visualization import visualize


import torch
import numpy as np
import os
torch.set_default_device('cuda')

obj_prompt, env_prompt, name = "a corner reflector spinning", "", "test"



if name is None:
    name = f"output_{int(time.time())}"

output_dir = os.path.join("output", name)
os.makedirs(output_dir, exist_ok=True)

body_pir, body_aux = pathtracer.trace()

print(colored('Skipping environment generation as requested.', 'yellow'))
env_pir = None

print(colored('Generating the radar signal.', 'green'))
radar_frames = signal_generator.generate_signal_frames(body_pir, body_aux, env_pir, radar_config="models/TI1843_config.json")

print(colored('[RFGen] Saving the radar bin file. Shape {}'.format(radar_frames.shape), 'green'))
np.save(os.path.join(output_dir, 'radar_frames.npy'), radar_frames)

print(colored('[RFGen] Rendering the visualization.', 'green'))
torch.set_default_device('cpu')  # To avoid OOM
visualize.save_video(
    "models/TI1843_config.json", 
    os.path.join(output_dir, 'radar_frames.npy'), 
    os.path.join(output_dir, 'obj_diff.npz'), 
    os.path.join(output_dir, 'output.mp4'))
exit(0)


    