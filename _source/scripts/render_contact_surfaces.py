"""Apply native surface parameters; retain original displacement and mesh generation."""
import sys
sys.dont_write_bytecode=True
import bpy

FIELDS={'height_intensity':'bump_height_intensity_list','spread':'bumps_spread_list',
        'aspect_ratio':'bumps_aspect_ratio_list','rotation_deg':'bump_rotation_deg_list',
        'center_angle_deg':'bump_center_angle_deg_list','center_offset':'bump_center_offset_list'}

def apply_surfaces(palm,fingers,thumb,frame):
    p=frame['palm_surface']
    palm.pad_resolution_level=6
    palm.bump_type='gaussian';palm.bumps_number=len(p['spread']);palm.bump_max_height_intensity_mm=p['height_mm']
    for key,field in FIELDS.items():setattr(palm,field,p[key])
    for cfg,profiles in zip(fingers+[thumb],frame['finger_surfaces']+[frame['thumb_surfaces']]):
        cfg.pad_resolution_level=5
        cfg.bump_type_list=['gaussian']*len(profiles)
        cfg.bump_number_list=[len(p['spread']) for p in profiles]
        cfg.bump_max_height_intensity_mm_list=[p['height_mm'] for p in profiles]
        for key,field in FIELDS.items():setattr(cfg,field,[p[key] for p in profiles])

def surface_mesh_stats():
    result={}
    for obj in bpy.context.scene.objects:
        if obj.type!='MESH' or not obj.name.lower().startswith('viz_pad') or obj.hide_render:continue
        points=[v.co for v in obj.data.vertices]
        if not points:continue
        for polygon in obj.data.polygons:polygon.use_smooth=True
        result[obj.name]=dict(vertices=len(points),local_z_extent_mm=1000*(max(p.z for p in points)-min(p.z for p in points)))
    return result
