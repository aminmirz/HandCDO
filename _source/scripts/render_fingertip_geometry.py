"""Measure the generated fingertip meshes in each original chain's local axes."""
import sys
sys.dont_write_bytecode=True
import bpy
import numpy as np

def measure_tips(native):
    dimensions=[]
    for i,digit in enumerate(native.fingers+native.thumbs):
        name=f'finger_{i+1}' if i<len(native.fingers) else 'thumb_1'
        col=bpy.data.collections[name]
        meshes=[]
        for obj in col.all_objects:
            if obj.type!='MESH' or not obj.name.lower().startswith('viz_'):continue
            ancestor=obj
            while ancestor is not None:
                if ancestor.name.startswith('link_4'):
                    meshes.append(obj);break
                ancestor=ancestor.parent
        assert meshes,(name,[o.name for o in col.all_objects])
        points=[]
        transform=np.linalg.inv(digit.elements[-1].transformation)
        for obj in meshes:
            for v in obj.data.vertices:
                p=obj.matrix_world@v.co
                points.append((transform@np.array([1000*p.x,1000*p.y,1000*p.z,1.]))[:3])
        points=np.array(points)
        dimensions.append((points.max(axis=0)-points.min(axis=0)).tolist())
    return dict(native_tip_scales=[list(d.cfg.fingertip_scale_factor) for d in native.fingers+native.thumbs],
                tip_dimensions_local_mm=dimensions,
                tip_attachment_matrices=[d.elements[-1].transformation.tolist() for d in native.fingers+native.thumbs])
