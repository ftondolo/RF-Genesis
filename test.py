import argparse
from termcolor import colored
import time
from genesis.raytracing import pathtracer
from genesis.raytracing import signal_generator

from genesis.environment_diffusion import environemnt_diff
#from genesis.object_diffusion import object_diff
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

# UPDATED: pathtracer.trace() now returns 3 values instead of 2
print(colored('[RFGen] Generating PIRs, radar pointclouds, and depth pointclouds.', 'green'))
PIRs, radar_pointclouds, depth_pointclouds = pathtracer.trace()

print(colored('Skipping environment generation as requested.', 'yellow'))
env_pir = None

print(colored('Generating the radar signal.', 'green'))
# UPDATED: Use new variable names (PIRs instead of body_pir, radar_pointclouds instead of body_aux)
radar_frames = signal_generator.generate_signal_frames(PIRs, radar_pointclouds, env_pir, radar_config="models/TI1843_config.json")

print(colored('[RFGen] Saving the radar bin file. Shape {}'.format(radar_frames.shape), 'green'))
np.save(os.path.join(output_dir, 'radar_frames.npy'), radar_frames)

# NEW: Save depth pointclouds to file for visualization
print(colored('[RFGen] Saving depth pointclouds for visualization.', 'green'))
depth_pointclouds_path = os.path.join(output_dir, 'depth_pointclouds.npy')
np.save(depth_pointclouds_path, depth_pointclouds, allow_pickle=True)

print(colored('[RFGen] Rendering the visualization.', 'green'))
torch.set_default_device('cpu')  # To avoid OOM

# UPDATED: Pass file path to depth pointclouds instead of raw data
visualize.save_video(
    "models/TI1843_config.json", 
    os.path.join(output_dir, 'radar_frames.npy'),
    depth_pointclouds_path,  # Pass the file path, not the raw data
    os.path.join(output_dir, 'output.mp4'))
    
print(colored('[RFGen] Complete! Video saved to {}'.format(os.path.join(output_dir, 'output.mp4')), 'green'))
exit(0)
