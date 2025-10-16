import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import io
import cv2
from tqdm import tqdm
import mitsuba as mi

from genesis.raytracing.radar import Radar 
from genesis.visualization.pointcloud import PointCloudProcessCFG, frame2pointcloud,rangeFFT,dopplerFFT,process_pc

mi.set_variant('scalar_rgb')

# Load and cache the PLY mesh
def load_ply_mesh(ply_path):
    """Load mesh from PLY file"""
    mesh = mi.load_dict({
        'type': 'ply',
        'filename': ply_path
    })
    params = mi.traverse(mesh)
    vertices = np.array(params['vertex_positions']).reshape(-1, 3)
    faces = np.array(params['faces']).reshape(-1, 3).astype(int)
    return vertices, faces

def rotate_vertices(vertices, axis, angle_degrees):
    """Rotate vertices around an axis by angle in degrees"""
    angle_rad = np.radians(angle_degrees)
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' rotation formula
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Rotation matrix using Rodrigues' formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + sin_angle * K + (1 - cos_angle) * np.dot(K, K)
    
    # Apply rotation
    rotated_vertices = np.dot(vertices, R.T)
    return rotated_vertices

# Plotting Pointclouds
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

def draw_ply_mesh_on_axis(vertices, faces, ax, elev, azim, title, frame_idx):
    """Draw the PLY mesh with rotation applied - synced with pathtracer.py logic"""
    # Apply rotation matching pathtracer.py: angle = frame_idx * 1.8
    angle = frame_idx * 1.8
    rotated_vertices = rotate_vertices(vertices, [0, 1, 0], angle)
    
    # Create mesh collection
    mesh = []
    for face in faces:
        triangle = rotated_vertices[face]
        mesh.append(triangle)
    
    collection = Poly3DCollection(mesh, alpha=0.7, facecolor='cyan', edgecolor='black', linewidths=0.1)
    ax.add_collection3d(collection)
    
    # Set axis limits to show the mesh properly
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=16)

def draw_doppler_on_axis(radar_frame,pointcloud_cfg, ax):
    range_fft = rangeFFT(radar_frame,pointcloud_cfg.frameConfig)
    doppler_fft = dopplerFFT(range_fft,pointcloud_cfg.frameConfig)
    dopplerResultSumAllAntenna = np.sum(doppler_fft, axis=(0,1))
    ax.imshow(np.abs(dopplerResultSumAllAntenna))
    ax.set_title("Doppler FFT", fontsize=16)

def draw_combined(i,pointcloud_cfg,radar_frames,pointclouds, ply_vertices, ply_faces):
    radar_frame_id = i

    fig= plt.figure(figsize=(18, 6))

    # Left: Point clouds
    ax1 = fig.add_subplot(131, projection='3d')
    draw_poinclouds_on_axis(pointclouds[radar_frame_id],ax1, None,None,30,-30,"Point Clouds")

    # Middle: Doppler FFT
    ax2 = fig.add_subplot(132)
    draw_doppler_on_axis(radar_frames[radar_frame_id],pointcloud_cfg, ax2)
    
    # Right: PLY mesh (synced with pathtracer rotation)
    ax3 = fig.add_subplot(133, projection='3d')
    draw_ply_mesh_on_axis(ply_vertices, ply_faces, ax3, 30, -30, "PLY Mesh (Rotating)", radar_frame_id)

    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig) 
    return data


def save_video(radar_cfg_file, radar_frames_file, output_file, ply_path='models/trihedral.ply'):
    radar = Radar(radar_cfg_file)
    pointcloud_cfg = PointCloudProcessCFG(radar)
    radar_frames = np.load(radar_frames_file)
    
    # Load PLY mesh
    print("Loading PLY mesh...")
    ply_vertices, ply_faces = load_ply_mesh(ply_path)
    print(f"Loaded mesh with {len(ply_vertices)} vertices and {len(ply_faces)} faces")

    # Process the pointclouds
    pointclouds = []
    for frame in radar_frames:
        pc = process_pc(pointcloud_cfg, frame)
        pointclouds.append(pc)
    
    # Write the video with wider resolution for 3 plots
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = output_file
    out = cv2.VideoWriter(video_filename, fourcc, 30, (1800, 600))
    for i in tqdm(range(len(radar_frames)), desc="Rendering video frames"):
        frame = draw_combined(i,pointcloud_cfg,radar_frames,pointclouds, ply_vertices, ply_faces)
        rgb_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(rgb_data)
    out.release()
    print(f"Video saved to {video_filename}")
