import mitsuba as mi
import numpy as np
import cv2

mi.set_variant('cuda_ad_rgb')
from mitsuba.scalar_rgb import Transform4f as T

images = []

for i in range(720):
    scene = mi.load_dict({
        'type': 'scene',
        'integrator': {'type': 'path'},
        'sensor': {
            'type': 'perspective',
            'fov': 60,
            'to_world': mi.ScalarTransform4f.look_at([0, 0, 1], [0,0,0], [0,1,0]),
            'film': {'type': 'hdrfilm', 'width': 256, 'height': 256},
            'sampler': {'type': 'independent', 'sample_count': 32}
        },
        'mesh': {
            'type': 'ply',
            'filename': '/content/RF-Genesis/models/trihedral.ply',
            'to_world': T.scale([1, 1, 1]) @ T.rotate([0,0,1], 90) @ T.rotate([0,1,0], i * 1.0),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': { 'type': 'rgb', 'value': [0.8, 0.8, 0.8] }
            }
        },
        'floor': {
            'type': 'rectangle',
            'to_world': T.translate([0,0,0]) @ T.scale([10,10,10]),
            'bsdf': {
                'type': 'diffuse',
                'reflectance': { 'type': 'rgb', 'value': [0.3, 0.3, 0.3] }
            }
        },
        'light': {
		'type': 'spot',
                'cutoff_angle': 40,
                'to_world': T.look_at(
                                origin=(0, 0, 1),
                                target=(0, 0, 0),
                                up=(0, 1, 0)
                            ),
                'intensity': { 'type': 'rgb', 'value': [2.0, 2.0, 2.0] }
        },
        'ambient': {
            'type': 'constant',
            'radiance': { 'type': 'rgb', 'value': 0.25 }
        }
    })
    
    img = mi.render(scene)
    img_np = np.array(img)[:,:,:3]
    
    # Tone mapping
    img_np = img_np / (1 + img_np)
    img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    images.append(img_bgr)
    
    if (i+1) % 60 == 0: print(f"{i+1}/720 frames")

# Export video
h, w = images[0].shape[:2]
out = cv2.VideoWriter('depth.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (w,h))
for img in images:
    out.write(img)
out.release()
print("Video saved to depth.mp4")
