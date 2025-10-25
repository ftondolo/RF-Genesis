import drjit as dr
import mitsuba as mi
from mitsuba.scalar_rgb import Transform4f as T
import numpy as np
from . import smpl
import torch
from tqdm import tqdm
mi.set_variant('cuda_ad_rgb')
torch.set_default_device('cuda')


class RayTracer:
    def __init__(self) -> None:
        self.PIR_resolution = 128
        self.scene = mi.load_dict(get_deafult_scene(res = self.PIR_resolution))
        self.params_scene = mi.traverse(self.scene)
        self.body = None #smpl.get_smpl_layer()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.axis = [0, 1, 0]
        self.angle = 3.6

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

    def update_mesh_rotation(self, axis=[0, 1, 0], angle=3.6):
        # Get current vertices and convert to match th_verts format
        vertices_flat = self.params_scene['smpl.vertex_positions']
        vertices_np = np.array(vertices_flat).reshape(-1, 3)
        vertices = torch.from_numpy(vertices_np).float().cuda()  
        if len(vertices.shape)==3:
            if vertices.shape[0]==1:
                vertices=vertices[0]

        # Apply rotation
        ones = torch.ones(vertices.shape[0], 1).cuda()
        homogeneous_vertices = torch.cat([vertices, ones], dim=1)
        transform = mi.Transform4f.rotate(axis=axis, angle=angle)
        rotation_matrix = torch.from_numpy(np.array(transform.matrix)).float().cuda()
        transformed_vertices = torch.matmul(homogeneous_vertices, rotation_matrix.T)
        transformed_vertices_3d = transformed_vertices[:, :3]
        vertices_mi = mi.TensorXf(transformed_vertices_3d.cpu().numpy())

        # Update scene
        self.params_scene['smpl.vertex_positions'] = dr.ravel(vertices_mi)
        self.params_scene.update()

    def trace(self):
        ray = self.gen_rays()
        si = self.scene.ray_intersect(ray)
        
        # Debug: Check the size of the intersection results
        t_array = np.array(si.t)
        print(f"Debug: t_array shape = {t_array.shape}, expected = ({self.PIR_resolution*self.PIR_resolution},)")
        
        if t_array.size == 1:
            print("Error: Only got a single intersection value. Ray generation failed.")
            # Return zeros as fallback
            return np.zeros((self.PIR_resolution, self.PIR_resolution, 3)), np.zeros((self.PIR_resolution*self.PIR_resolution, 3))
        
        # Continue with normal processing
        t_array[t_array > 9999] = 0
        distance = t_array.reshape(self.PIR_resolution, self.PIR_resolution)
        
        intensity = mi.render(self.scene, spp=32)
        intensity = np.array(intensity)[:, :, 0]
        velocity = np.zeros((self.PIR_resolution, self.PIR_resolution))
        
        PIR = np.stack([distance, intensity, velocity], axis=2)
        pointclouds = np.array(si.p)
        return PIR, pointclouds
    


def get_deafult_scene(res = 512):
    integrator = mi.load_dict({
        'type': 'direct',
        })

    sensor = mi.load_dict({
            'type': 'perspective',
            'to_world': T.look_at(
                            origin=(0, 2, 4),
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
                'filename': '/content/RF-Genesis/models/trihedral.ply',
                'to_world' : T.translate([0, 0, 0]),
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



def trace(motion_filename=None, rotation_axis=[0,1,0], angle=3.6):
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
