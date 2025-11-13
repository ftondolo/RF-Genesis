import argparse
from genesis.visualization import visualize


import torch
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple script that greets a user.")
    parser.add_argument("name", type=str, help="The name to greet")

    args = parser.parse_args()

    torch.set_default_device('cuda')

    output_dir = os.path.join("output", args.name)

    depth_pointclouds_path = os.path.join(output_dir, 'depth_pointclouds.npy')

    torch.set_default_device('cpu')  # To avoid OOM

    # UPDATED: Pass file path to depth pointclouds instead of raw data
    visualize.save_video(
        "models/TI1843_config.json", 
        os.path.join(output_dir, 'radar_frames.npy'),
        depth_pointclouds_path,  # Pass the file path, not the raw data
        os.path.join(output_dir, 'output.mp4'))
        
    print('[RFGen] Complete!')
    exit(0)
