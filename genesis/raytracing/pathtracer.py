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
        self.scene = mi.load_dict(get_deafult_scene(res=self.PIR_resolution))
        self.params_scene = mi.traverse(self.scene)

        # Store original vertices to avoid accumulation
        original_verts = self.params_scene['smpl.vertex_positions']
        dr.eval(original_verts)
        self.original_vertices = np.array(dr.detach(original_verts)).copy()

        self.body = None #smpl.get_smpl_layer()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.axis = [0, 1, 0]
        self.angle = 1.0
        self.cumulative_angle = -1.0  # Track cumulative rotation

    def gen_rays(self):
        """Generate rays for all pixels in the film"""
        import drjit as dr
        
        sensor = self.scene.sensors()[1]  # Main sensor (128x128) is at index 1
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

    def update_mesh_rotation(self):
        # Update cumulative rotation angle
        self.cumulative_angle += self.angle

        # Start from original vertices (prevents accumulation)
        vertices_np = self.original_vertices.reshape(-1, 3)

        # Create rotation matrix with cumulative angle
        rotation_transform = mi.Transform4f.rotate(axis=self.axis, angle=self.cumulative_angle)
        rotation_matrix = np.array(rotation_transform.matrix)[:3, :3]

        # Apply rotation to original vertices
        rotated_vertices = vertices_np @ rotation_matrix.T

        # Flatten and convert back to DrJit array
        rotated_flat = rotated_vertices.flatten()

        # Create new DrJit array - use dr.cuda.ad.Float to match the variant
        from drjit.cuda.ad import Float as CudaFloat
        result = CudaFloat(rotated_flat)

        # Update scene
        self.params_scene['smpl.vertex_positions'] = result
        self.params_scene.update()

    def trace(self):
        ray = self.gen_rays()
        si = self.scene.ray_intersect(ray)
        
        # Use main sensor (128x128) for rendering, not default sensor
        main_sensor = self.scene.sensors()[1]
        intensity = mi.render(self.scene, sensor=main_sensor, spp=32)

        t = np.array(si.t)
        t[t > 99999] = 0
        distance = t.reshape(self.PIR_resolution, self.PIR_resolution)
        intensity = np.array(intensity)[:, :, 0]
        velocity = np.zeros((self.PIR_resolution, self.PIR_resolution))

        PIR = np.stack([distance, intensity, velocity], axis=2)
        pointclouds = np.array(si.p)
        return PIR, pointclouds
    
    def get_depth_pointcloud(self):
        """
        Generate high-resolution depth pointcloud from dedicated depth sensor.
        This is separate from the radar pipeline and used purely for visualization.
        
        Returns:
            depth_pointcloud: numpy array of shape (N, 3) with valid 3D points
        """
        import drjit as dr
        
        # Use the depth sensor (256x256) at index 0
        depth_sensor = self.scene.sensors()[0]
        film = depth_sensor.film()
        sampler = depth_sensor.sampler()
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
        
        # Sample rays from depth sensor
        rays = depth_sensor.sample_ray(
            time=sampler.next_1d(),
            sample1=sampler.next_1d(), 
            sample2=pos,
            sample3=sampler.next_1d()
        )
        
        # Handle tuple return
        if isinstance(rays, tuple):
            rays = rays[0]
        
        # Intersect rays with scene
        si = self.scene.ray_intersect(rays)
        
        # Get 3D intersection points
        depth_pointcloud = np.array(si.p)
        
        # Filter out invalid points (no intersection or too far)
        distances = np.array(si.t)
        valid_mask = distances < 1000  # Filter points beyond 100 units
        
        # Reshape and filter
        depth_pointcloud = depth_pointcloud.reshape(-1, 3)
        depth_pointcloud = depth_pointcloud[valid_mask]
        
        return depth_pointcloud
    


def get_deafult_scene(res):
    integrator = mi.load_dict({
        'type': 'direct',
        })

    # Main sensor for radar/PIR pipeline
    sensor = mi.load_dict({
            'type': 'perspective',
            'to_world': T.look_at(
                            origin=(1, 0, 0),
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

    # Dedicated depth sensor for visualization (higher resolution)
    depth_sensor = mi.load_dict({
            'type': 'perspective',
            'to_world': T.look_at(
                            origin=(1, 0, 0),
                            target=(0, 0, 0),
                            up=(0, 1, 0)
                        ),
            'fov': 60,
            'film': {
                'type': 'hdrfilm',
                'width': 256,
                'height': 256,
                'rfilter': { 'type': 'gaussian' },
                'sample_border': True,
                'pixel_format': 'luminance',
                'component_format': 'float32',
            },
            'sampler':{
                'type': 'independent',
                'sample_count': 1,
                'seed':43
            },
        })


    default_scene ={
            'type': 'scene',
            'integrator': integrator,
            'depth_sensor': depth_sensor,  # Depth sensor first (index 0)
            'sensor': sensor,  # Main sensor second (index 1)
            
            'while':{
                'type':'diffuse',
                'reflectance': { 'type': 'rgb', 'value': (0.8, 0.8, 0.8) }, 
            },
            'smpl':{
                'type': 'ply',
                'filename': '/content/RF-Genesis/models/trihedral.ply',
                'to_world' : T.scale(1).translate([0, 0, 0]).rotate(axis=[0, 0, 1], angle=45),
                "mybsdf": {
                    "type": "ref",
                    "id": "while"
                },
            },

            'tx':{
                'type': 'spot',
                'cutoff_angle': 40,
                'to_world': T.look_at(
                                origin=(1, 0, 0),
                                target=(0, 0, 0),
                                up=(0, 1, 0)
                            ),
                'intensity': 1000.0,
            }

        }
    return default_scene



def trace(motion_filename=None):
    """
    Trace rays through SMPL body motion sequence.

    Args:
        motion_filename: Path to .npz file containing SMPL motion data
        rotation_axis: Optional axis for mesh rotation [x, y, z]
        angle: Rotation angle per frame in degrees
        depth_resolution: Resolution for depth sensor (for visualization)

    Returns:
        PIRs: List of PIR tensors (for radar pipeline)
        pointclouds: List of pointcloud tensors (for radar pipeline)
        depth_pointclouds: List of depth pointclouds (for visualization)
    """

    raytracer = RayTracer()
    PIRs = []
    pointclouds = []
    depth_pointclouds = []
    total_motion_frames = 721

    for frame_idx in tqdm(range(0, total_motion_frames), desc="Rendering PIRs"):
        raytracer.update_mesh_rotation()

        # Radar pipeline (128x128)
        PIR, pc = raytracer.trace()
        PIRs.append(torch.from_numpy(PIR).cuda())
        pointclouds.append(torch.from_numpy(pc).cuda())
        
        # Depth pointcloud for visualization (higher resolution)
        depth_pc = raytracer.get_depth_pointcloud()
        depth_pointclouds.append(depth_pc)  # Keep as numpy for easy saving

    return PIRs, pointclouds, depth_pointclouds
