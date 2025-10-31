import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import io
import cv2
from tqdm import tqdm

from genesis.raytracing.radar import Radar 
from genesis.visualization.pointcloud import PointCloudProcessCFG, frame2pointcloud,rangeFFT,dopplerFFT,process_pc


def draw_depth_pointcloud(depth_pc, ax, elev, azim):
    if len(depth_pc) == 0:
        ax.text(0, 0, 0, 'No valid points', fontsize=12)
        return ax
    
    colors = depth_pc[:, 2]
    ax.scatter(depth_pc[:, 0], depth_pc[:, 1], depth_pc[:, 2], 
              c=colors, cmap=plt.cm.viridis, s=2, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0, 6)
    ax.set_zlim(-0.5, 2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title('Depth Pointcloud', fontsize=20)
    return ax


def draw_poinclouds_on_axis(pc,ax, tx,rx,elev,azim,title):
    pc = np.transpose(pc)
    ax.scatter(-pc[0], pc[1], pc[2], c=pc[4], cmap=plt.hot())
    if tx is not None:
        ax.scatter(tx[:,0], tx[:,2], tx[:,1], c="green", s= 50, marker =',', cmap=plt.hot())
    if rx is not None:
        ax.scatter(rx[:,0], rx[:,2], rx[:,1], c="orange", s= 50, marker =',', cmap=plt.hot())
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0, 6)
    ax.set_zlim(-0.5, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=20)

def draw_doppler_on_axis(radar_frame,pointcloud_cfg, ax):
    range_fft = rangeFFT(radar_frame,pointcloud_cfg.frameConfig)
    doppler_fft = dopplerFFT(range_fft,pointcloud_cfg.frameConfig)
    dopplerResultSumAllAntenna = np.sum(doppler_fft, axis=(0,1))
    ax.imshow(np.abs(dopplerResultSumAllAntenna))
    ax.set_title("Doppler FFT", fontsize=20)

def draw_combined(i, pointcloud_cfg, radar_frames, radar_pointclouds, depth_pointclouds):
    radar_frame_id = int(i/3)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(131, projection='3d')
    draw_depth_pointcloud(depth_pointclouds[radar_frame_id], ax1, elev=30, azim=240)

    ax2 = fig.add_subplot(132, projection='3d')
    draw_poinclouds_on_axis(radar_pointclouds[radar_frame_id], ax2, None, None, 30, 240, "Radar Pointcloud")

    ax3 = fig.add_subplot(133)
    draw_doppler_on_axis(radar_frames[radar_frame_id], pointcloud_cfg, ax3)

    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig) 
    return data


def save_video(radar_cfg_file, radar_frames_file, depth_pointclouds_file, output_file):
    radar = Radar(radar_cfg_file)
    pointcloud_cfg = PointCloudProcessCFG(radar)
    radar_frames = np.load(radar_frames_file)
    depth_pointclouds = np.load(depth_pointclouds_file, allow_pickle=True)

    radar_pointclouds = []
    for frame in radar_frames:
        pc = process_pc(pointcloud_cfg, frame)
        radar_pointclouds.append(pc)
    
    num_radar_frames = len(radar_frames)
    num_video_frames = num_radar_frames * 3
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 30, (1200, 600))
    
    for i in tqdm(range(num_video_frames)):
        frame = draw_combined(i, pointcloud_cfg, radar_frames, radar_pointclouds, depth_pointclouds)
        rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(rgb_data)
    
    out.release()
    print(f"Video saved to: {output_file}")
