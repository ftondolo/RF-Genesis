import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from tqdm import tqdm

from genesis.raytracing.radar import Radar 
from genesis.visualization.pointcloud import PointCloudProcessCFG, frame2pointcloud,rangeFFT,dopplerFFT,process_pc


def draw_depth_camera_pointcloud_on_axis(depth_pc, ax, elev, azim, title, colormap='viridis'):
    """Draw the depth camera pointcloud from Mitsuba scene
    
    Args:
        depth_pc: pointcloud from Mitsuba depth camera (N, 3) array
        ax: matplotlib 3D axis
        elev: elevation angle for viewing
        azim: azimuth angle for viewing  
        title: subplot title
        colormap: colormap for point coloring based on depth
    """
    if isinstance(depth_pc, torch.Tensor):
        depth_pc = depth_pc.cpu().numpy()
    
    # Reshape if needed
    if len(depth_pc.shape) == 2 and depth_pc.shape[1] == 3:
        points = depth_pc
    else:
        # Flatten if it's from the PIR resolution (e.g., 128x128x3)
        points = depth_pc.reshape(-1, 3)
    
    # Filter out points at infinity or invalid points
    valid_mask = (np.abs(points[:, 0]) < 1000) & (np.abs(points[:, 1]) < 1000) & (np.abs(points[:, 2]) < 1000)
    valid_mask &= ~np.isnan(points).any(axis=1)
    valid_mask &= ~np.isinf(points).any(axis=1)
    points = points[valid_mask]
    
    if len(points) > 0:
        # Subsample if too many points for visualization
        if len(points) > 10000:
            indices = np.random.choice(len(points), 10000, replace=False)
            points = points[indices]
        
        # Color by depth (z-coordinate)
        colors = points[:, 2]
        
        # Plot the pointcloud
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=colors, cmap=colormap, s=1, alpha=0.6)
        
        # Add colorbar for depth
        # plt.colorbar(scatter, ax=ax, label='Depth')
    
    # Set reasonable axis limits based on camera view
    limit = 2.0  # Adjust based on your scene scale
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit) 
    ax.set_zlim(-limit, limit)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=16)


def draw_radar_pointcloud_on_axis(pc, ax, tx, rx, elev, azim, title):
    """Draw radar-processed pointclouds"""
    pc = np.transpose(pc)
    ax.scatter(-pc[0], pc[1], pc[2], c=pc[4], cmap=plt.hot())
    if tx is not None:
        ax.scatter(tx[:,0], tx[:,2], tx[:,1], c="green", s=50, marker=',', cmap=plt.hot())
    if rx is not None:
        ax.scatter(rx[:,0], rx[:,2], rx[:,1], c="orange", s=50, marker=',', cmap=plt.hot())
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 6)
    ax.set_zlim(-0.5, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=20)


def draw_doppler_on_axis(radar_frame, pointcloud_cfg, ax):
    """Draw Doppler FFT heatmap"""
    range_fft = rangeFFT(radar_frame, pointcloud_cfg.frameConfig)
    doppler_fft = dopplerFFT(range_fft, pointcloud_cfg.frameConfig)
    dopplerResultSumAllAntenna = np.sum(doppler_fft, axis=(0,1))
    ax.imshow(np.abs(dopplerResultSumAllAntenna))
    ax.set_title("Doppler FFT", fontsize=16)
    ax.set_xlabel("Range Bins")
    ax.set_ylabel("Doppler Bins")


def draw_combined(i, pointcloud_cfg, radar_frames, radar_pointclouds, depth_camera_pointclouds):
    """Draw combined visualization with depth camera pointcloud, Doppler FFT, and radar pointcloud"""
    
    # Map radar frame index to depth camera frame index
    # Radar: ~67 frames at 10 FPS
    # Depth camera: 200 frames at 30 FPS
    # Mapping: depth_frame = radar_frame * 3
    depth_frame_idx = min(i * 3, len(depth_camera_pointclouds) - 1)
    
    fig = plt.figure(figsize=(18, 6))
    
    # Left: Depth camera pointcloud from Mitsuba
    ax1 = fig.add_subplot(131, projection='3d')
    draw_depth_camera_pointcloud_on_axis(
        depth_camera_pointclouds[depth_frame_idx], 
        ax1, 20, -60, 
        f"Depth Camera Pointcloud (Frame {depth_frame_idx})"
    )
    
    # Middle: Doppler FFT
    ax2 = fig.add_subplot(132)
    draw_doppler_on_axis(radar_frames[i], pointcloud_cfg, ax2)
    
    # Right: Radar pointcloud
    ax3 = fig.add_subplot(133, projection='3d')
    draw_radar_pointcloud_on_axis(
        radar_pointclouds[i], 
        ax3, None, None, 30, -30, 
        f"Radar Pointcloud (Frame {i})"
    )
    
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data


def save_video(radar_cfg_file, radar_frames_file, depth_camera_pointclouds, output_file):
    """Save video with depth camera pointclouds instead of PLY mesh
    
    Args:
        radar_cfg_file: path to radar configuration JSON
        radar_frames_file: path to saved radar frames .npy file
        depth_camera_pointclouds: list of depth camera pointclouds from Mitsuba
        output_file: output video file path
    """
    radar = Radar(radar_cfg_file)
    pointcloud_cfg = PointCloudProcessCFG(radar)
    radar_frames = np.load(radar_frames_file)
    
    # Convert depth camera pointclouds to numpy if they're torch tensors
    if isinstance(depth_camera_pointclouds, list) and len(depth_camera_pointclouds) > 0:
        if isinstance(depth_camera_pointclouds[0], torch.Tensor):
            depth_camera_pointclouds = [pc.cpu().numpy() for pc in depth_camera_pointclouds]
    
    print(f"Processing {len(radar_frames)} radar frames and {len(depth_camera_pointclouds)} depth camera frames")
    
    # Process the radar pointclouds
    radar_pointclouds = []
    for frame in radar_frames:
        pc = process_pc(pointcloud_cfg, frame)
        radar_pointclouds.append(pc)
    
    # Write the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 10, (1800, 600))  # 10 FPS for radar frame rate
    
    for i in tqdm(range(len(radar_frames)), desc="Rendering video frames"):
        frame = draw_combined(i, pointcloud_cfg, radar_frames, radar_pointclouds, depth_camera_pointclouds)
        rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(rgb_data)
    
    out.release()
    print(f"Video saved to {output_file}")
