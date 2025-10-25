import drjit as dr
import numpy as np
from . import smpl
import torch
from tqdm import tqdm
import mitsuba as mi

# Import scalar Transform4f for scene definition
from mitsuba.scalar_rgb import Transform4f as T

# IMPORTANT: Set variant AFTER all imports
# Importing from mitsuba.scalar_rgb switches the variant, so we must reset it
mi.set_variant('cuda_ad_rgb')
torch.set_default_device('cuda')


class RayTracer:
    def __init__(self) -> None:
        # Force cuda variant right before scene loading
        mi.set_variant('cuda_ad_rgb')

        self.PIR_resolution = 128
        self.scene = mi.load_dict(get_deafult_scene(res = self.PIR_resolution))
        self.params_scene = mi.traverse(self.scene)
        self.body = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Store original vertices to avoid accumulation
        original_verts = self.params_scene['smpl.vertex_positions']
        dr.eval(original_verts)
        self.original_vertices = np.array(dr.detach(original_verts)).copy()

        self.body = None #smpl.get_smpl_layer()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.axis = [0, 1, 0]
        self.angle = 3.6
        self.cumulative_angle = 0.0  # Track cumulative rotation

    def gen_rays(self):
        """Generate rays for all pixels in the film"""
        import drjit as dr
        
        sensor = self.scene.sensors()[0]
        film = sensor.film()
        sampler = sensor.sampler()
        film_size = film.crop_size()
        
        # Total number of pixels
        pixel_count = int(film_size[0] * film_size[1])
        
        # Seed sampler
        sampler.seed(0, pixel_count)
        
        # Generate pixel indices
        idx = dr.arange(mi.UInt32, pixel_count)
        
        # Convert to 2D coordinates
        x = idx % int(film_size[0])
        y = idx // int(film_size[0])
        
        # Normalize to [0,1] and add 0.5 to sample pixel centers
        pos = mi.Point2f(
            (mi.Float(x) + 0.5) / film_size[0],
            (mi.Float(y) + 0.5) / film_size[1]
        )
        
        # Sample rays - the result is already a ray bundle, not a tuple
        rays = sensor.sample_ray(
            time=sampler.next_1d(),
            sample1=sampler.next_1d(), 
            sample2=pos,
            sample3=sampler.next_1d()
        )
        
        # Check if it's returning a tuple and extract rays if needed
        if isinstance(rays, tuple):
            rays = rays[0]
        
        return rays
    
    def update_pose(self,pose_params, shape_params, translation= None):
        
        pose_params = torch.tensor(pose_params).view(1, -1)
        shape_params = torch.tensor(shape_params).view(1, -1)

        if translation is not None:
            transform = mi.Transform4f.translate(translation)
            transform = torch.tensor(transform.matrix).squeeze()


        vertices_mi=smpl.call_smpl_layer(pose_params,shape_params,self.body,need_face=False,transform=transform)
        
        self.params_scene['smpl.vertex_positions'] = dr.ravel(vertices_mi)
        self.params_scene.update()
    
    def update_sensor(self,origin, target):
        transform = mi.Transform4f.look_at(
                            origin=origin,
                            target=target,
                            up=(0, 1, 0)
                        )
        self.params_scene['sensor.to_world'] = transform
        self.params_scene['tx.to_world'] = transform
        self.params_scene.update()

    def update_mesh_rotation(self, axis=[0, 1, 0], angle=0):
        """
        Rotate the mesh around a given axis by a given angle.

        Args:
            axis: Rotation axis as [x, y, z]
            angle: Rotation angle in degrees
        """
        # Get current vertices from the scene (flat buffer: x1,y1,z1,x2,y2,z2,...)
        current_vertices = self.params_scene['smpl.vertex_positions']

        # Convert to numpy for easier manipulation
        vertices_np = np.array(current_vertices).reshape(-1, 3)

        # Create rotation matrix from axis and angle
        rotation_transform = mi.Transform4f.rotate(axis=axis, angle=angle)
        rotation_matrix = np.array(rotation_transform.matrix)[:3, :3]

        # Apply rotation to each vertex
        rotated_vertices_np = vertices_np @ rotation_matrix.T

        # Convert back to flat buffer format
        rotated_flat = rotated_vertices_np.flatten()

        # Convert back to DrJit array with same type as original
        self.params_scene['smpl.vertex_positions'] = type(current_vertices)(rotated_flat)
        self.params_scene.update()

    def trace(self):
        ray = self.gen_rays()
        si = self.scene.ray_intersect(ray)                   # ray intersection
        intensity = mi.render(self.scene,spp=32)

        # Ensure DrJit arrays are evaluated before conversion
        dr.eval(si.t)
        dr.eval(si.p)

        # Convert to numpy and handle the flattened array
        t = np.array(si.t, copy=False)
        pointclouds = np.array(si.p, copy=False)

        # Check if we got the expected number of samples
        expected_size = self.PIR_resolution * self.PIR_resolution
        if t.size != expected_size:
            print(f"Warning: Expected {expected_size} samples but got {t.size}")
            # If t is scalar, broadcast it to the expected size
            if t.size == 1:
                t = np.full(expected_size, t.item())

            # Also fix pointclouds shape if needed
            if pointclouds.size == 3:  # Single point (x, y, z)
                # Create a default pointcloud array with zeros
                pointclouds = np.zeros((expected_size, 3), dtype=pointclouds.dtype)

        # Ensure pointclouds is reshaped correctly
        if pointclouds.ndim == 1 and pointclouds.size == expected_size * 3:
            pointclouds = pointclouds.reshape(expected_size, 3)
        elif pointclouds.shape[0] != expected_size:
            print(f"Warning: Pointcloud shape mismatch. Expected ({expected_size}, 3) but got {pointclouds.shape}")
            pointclouds = np.zeros((expected_size, 3), dtype=pointclouds.dtype)

        t[t>9999]=0
        distance = t.reshape(self.PIR_resolution,self.PIR_resolution)
        intensity = np.array(intensity)[:,:,0]
        velocity = np.zeros((self.PIR_resolution,self.PIR_resolution))  # the velocity is zero for this static frame,
                                                                        # but will be calculated later by calculating the difference between two frames

        PIR = np.stack([distance,intensity,velocity],axis=2)
        return PIR, pointclouds
    


def get_deafult_scene(res = 512):
    import os

    # Get the absolute path to the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    ply_path = os.path.join(project_root, 'models', 'trihedral.ply')

    integrator = mi.load_dict({
        'type': 'direct',
        })

    sensor = mi.load_dict({
            'type': 'perspective',
            'to_world': T.look_at(
                            origin=(0, 0, 3),
                            target=(0, 0, 0),
                            up=(0, 1, 0)
                        ),
            'fov': 60,
            'film': {
                'type': 'hdrfilm',
                'width': res,
                'height': res,
                'rfilter': { 'type': 'gaussian' },
                'sample_border': True,
                'pixel_format': 'luminance',
                'component_format': 'float32',
            },
            'sampler':{
                'type': 'independent',
                'sample_count': 1,
                'seed':42
            },
        })


    default_scene ={
            'type': 'scene',
            'integrator': integrator,
            'sensor': sensor,

            'while':{
                'type':'diffuse',
                'reflectance': { 'type': 'rgb', 'value': (0.8, 0.8, 0.8) },
            },
            'smpl':{
                'type': 'ply',
                'filename': ply_path,
                "mybsdf": {
                    "type": "ref",
                    "id": "while"
                },
            },

            'tx':{
                'type': 'spot',
                'cutoff_angle': 40,
                'to_world': T.look_at(
                                origin=(2, 3, 4),
                                target=(0, 0, 0),
                                up=(0, 1, 0)
                            ),
                'intensity': 1000.0,
            }

        }
    return default_scene



def trace(motion_filename=None, rotation_axis=[0, 1, 0], angle=3.6):
    """
    Trace rays through SMPL body motion sequence.

    Args:
        motion_filename: Path to .npz file containing SMPL motion data
        rotation_axis: Optional axis for mesh rotation [x, y, z]
        rotation_angles: Optional list of rotation angles (in degrees) for each frame

    Returns:
        PIRs: List of PIR tensors
        pointclouds: List of pointcloud tensors
    """
    #smpl_data = np.load(motion_filename, allow_pickle=True)
    #root_translation = smpl_data['root_translation']
    #max_distance = np.max(root_translation[:,2])+2
    #body_offset = np.array([0,1,3])
    sensor_origin = np.array([0,0,0])
    sensor_target = np.array([0,0,-5])

    raytracer = RayTracer()
    PIRs = []
    pointclouds = []
    total_motion_frames = 200

    for frame_idx in tqdm(range(0, total_motion_frames), desc="Rendering PIRs"):
        raytracer.update_mesh_rotation(axis=rotation_axis, angle=angle)

        PIR, pc = raytracer.trace()
        PIRs.append(torch.from_numpy(PIR).cuda())
        pointclouds.append(torch.from_numpy(pc).cuda())

    # pointclouds = torch.stack(pointclouds, dim=0)
    return PIRs, pointclouds
