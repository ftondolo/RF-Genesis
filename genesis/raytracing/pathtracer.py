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
        self.base_vertices = self._load_ply_vertices('/content/RF-Genesis/models/trihedral.ply')
    
    def _load_ply_vertices(self, ply_path):
        """Load vertices from PLY file"""
        mesh = mi.load_dict({
            'type': 'ply',
            'filename': ply_path
        })
        params = mi.traverse(mesh)
        vertices = params['vertex_positions']
        return vertices
        
    def gen_rays(self):  
        sensor = self.scene.sensors()[0]
        film = sensor.film()
        sampler = sensor.sampler()
        film_size = film.crop_size()
        spp = 1
        total_sample_count = dr.prod(film_size) * spp
        if sampler.wavefront_size() != total_sample_count:
            sampler.seed(0, total_sample_count)

        pos = dr.arange(mi.UInt32, total_sample_count)
        pos //= spp
        scale = mi.Vector2f(1.0 / film_size[0], 1.0 / film_size[1])
        pos = mi.Vector2f(mi.Float(pos  % int(film_size[0])),
                    mi.Float(pos // int(film_size[0])))
        rays, weights = sensor.sample_ray_differential(
            time=0,
            sample1=sampler.next_1d(),
            sample2=pos * scale,
            sample3=0
        )
        return rays
    
    def update_mesh_rotation(self, axis, angle):
        """Rotate the mesh by applying transform to base vertices"""
        # Create rotation transform
        transform = mi.Transform4f.rotate(axis=axis, angle=angle)
        
        # Apply transform to base vertices
        vertices_flat = dr.ravel(self.base_vertices)
        num_vertices = len(vertices_flat) // 3
        
        # Reshape to (N, 3) for transformation
        vertices_reshaped = mi.Point3f(
            vertices_flat[0::3],
            vertices_flat[1::3],
            vertices_flat[2::3]
        )
        
        # Apply rotation
        rotated_vertices = transform @ vertices_reshaped
        
        # Update scene
        self.params_scene['smpl.vertex_positions'] = dr.ravel(rotated_vertices)
        self.params_scene.update()
    
    def update_pose(self,pose_params, shape_params, translation= None):
        if pose_params is None:  # Skip SMPL, just apply transform
            if translation is not None:
                transform = mi.Transform4f.translate(translation)
                self.params_scene['smpl.to_world'] = transform
                self.params_scene.update()
            return
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
    
    def trace(self):
        ray = self.gen_rays()
        si = self.scene.ray_intersect(ray)                   # ray intersection
        intensity = mi.render(self.scene,spp=32)
        t= si.t
        t[t>9999]=0
        distance = np.array(t).reshape(self.PIR_resolution,self.PIR_resolution)
        intensity = np.array(intensity)[:,:,0]
        velocity = np.zeros((self.PIR_resolution,self.PIR_resolution))  # the velocity is zero for this static frame, 
                                                                        # but will be calculated later by calculating the difference between two frames
        
        PIR = np.stack([distance,intensity,velocity],axis=2)
        pointclouds = np.array(si.p)        # We save the points here for faster calculation, it can be calculated from the PIR's distance + sensor's intrinsic metrix
        return PIR, pointclouds
    


def get_deafult_scene(res = 512):
    integrator = mi.load_dict({
        'type': 'direct',
        })

    sensor = mi.load_dict({
            'type': 'perspective',
            'to_world': T.look_at(
                            origin=(0, 1, 3),
                            target=(0, 1, 0),
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
                "mybsdf": {
                    "type": "ref",
                    "id": "while"
                },
            },

            'tx':{
                'type': 'spot',
                'cutoff_angle': 40,
                'to_world': T.look_at(
                                origin=(0, 0, 3),
                                target=(0, 0, 0),
                                up=(0, 1, 0)
                            ),
                'intensity': 1000.0,
            }

        }
    return default_scene



def trace(motion_filename=None):
    if motion_filename:
        smpl_data = np.load(motion_filename, allow_pickle=True)
        root_translation = smpl_data['root_translation']
        max_distance = np.max(root_translation[:,2])+2
        total_motion_frames = len(root_translation)
        
    total_motion_frames = 200
    max_distance = 2
    
    body_offset = np.array([0,1,3])
    sensor_origin = np.array([0,0,0])
    sensor_target = np.array([0,0,-5])

    raytracer = RayTracer()
    PIRs = []
    pointclouds = []

    for frame_idx in tqdm(range(0, total_motion_frames),desc="Rendering PLY PIRs"):
        angle = frame_idx * 1.8  
        raytracer.update_mesh_rotation(axis=[0, 1, 0], angle=angle)
        #raytracer.update_pose(smpl_data['pose'][frame_idx], smpl_data['shape'][0], np.array(root_translation[frame_idx]) -  body_offset)
        PIR, pc = raytracer.trace()
        PIRs.append(torch.from_numpy(PIR).cuda())
        pointclouds.append(torch.from_numpy(pc).cuda())

    # pointclouds = torch.stack(pointclouds, dim=0)
    return PIRs, pointclouds
