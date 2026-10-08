"""Keep the approved native thumb orbit, then add three native fingers."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_thumb_orbit import *

OUT=ROOT/'preview/thumb-assembly'

def plan():
    orbit=json.loads((ROOT/'preview/thumb-orbit/plan.json').read_text())
    fingers=json.loads((ROOT/'preview/finger-assembly/plan.json').read_text())
    frames=[]
    for source in range(0,181,2):
        frames.append(dict(index=len(frames),stage=1,orbit_source=source,
                           thumb_location=orbit['frames'][source]['location'],locations=[],identities=[]))
    for original in fingers['frames'][91:]:
        pairs=[(identity,location) for identity,location in zip(original['identities'],original['locations']) if identity!=0]
        frames.append(dict(index=len(frames),stage=original['stage'],thumb_location=frames[90]['thumb_location'],
                           locations=[p[1] for p in pairs],identities=[p[0] for p in pairs]))
    schedule=dict(frames=frames,order=[0]*6+list(range(184))+[183]*20,fps=30,duration_seconds=7,
                  source_hashes=orbit['source_hashes'],thumb_code=THUMB_CODE,finger_code='2-1-1')
    assert len(frames)==184 and len(schedule['order'])==210
    (OUT/'plan.json').write_text(json.dumps(schedule,indent=2))

def generate_frame(f,temp):
    cfg,curve=palm_config(f['thumb_location'])
    cfg.thumb_angle_deg_list[0]+=f.get('thumb_angle_delta_deg',0.)
    n=len(f['locations']);cfg.finger_number=n
    cfg.finger_location_list=f['locations']
    cfg.finger_points=np.array([cfg._curve_point_at(curve,t) for t in f['locations']]).reshape(n,2)
    cfg.finger_angle_deg_list=f.get('angles_deg',[0.]*n)
    cfg.finger_base_normal_offset_mm_list=f.get('normal_offsets_mm',[0.]*n)
    cfg.finger_base_side_offset_mm_list=[0.]*n
    codes=f.get('finger_codes',['2-1-1']*n)
    thumb_code=f.get('requested_thumb_code',THUMB_CODE)
    fingers=[FingerConfig(type='finger',data='fixed',code=codes[i],id=i) for i in range(n)]
    thumb=FingerConfig(type='thumb',data='fixed',code=thumb_code,id=0)
    if 'finger_tip_scales' in f:
        for digit,scale in zip(fingers,f['finger_tip_scales']):digit.fingertip_scale_factor=tuple(scale)
        thumb.fingertip_scale_factor=tuple(f['thumb_tip_scale'])
    if 'palm_surface' in f:
        from render_contact_surfaces import apply_surfaces
        apply_surfaces(cfg,fingers,thumb,f)
    if 'finger_link_lengths_mm' in f:
        for digit,lengths in zip(fingers,f['finger_link_lengths_mm']):
            digit.link_added_length_mm_list=list(lengths)
        thumb.link_added_length_mm_list=list(f['thumb_link_lengths_mm'])
    hc=HandConfig();hc.collision_mesh=False
    native=Hand(cfg,fingers,[thumb],hc,root_dir=str(temp))
    data=native.save_assembly_data();palm=native.save_palm_data()
    assert data and palm and len(native.fingers)==n and len(native.thumbs)==1
    addon._clear_generated_hand();addon._clear_template_objects()
    appended=addon._append_all_from_blend(str(GEN/'blender/components.blend'))
    assembly.FingerAssembly.from_existing_scene(data,appended).run_assembly()
    assembly.PalmMesh(palm).generate();assembly.assign_viz_materials();assembly.apply_origin_offset(palm)
    addon._clear_template_objects();assembly.shade_all_meshes_flat();addon._apply_hand_visibility(bpy.context)
    # Preserve finger colors as their location-sorted native indices change.
    palette=['t','f3','f2','f1']
    missing=[name for name in palette if bpy.data.materials.get(name) is None]
    if missing:
        with bpy.data.libraries.load(str(GEN/'blender/components.blend'),link=False) as (src,dst):dst.materials=missing
    for col in bpy.data.collections['hand'].children:
        if not col.name.startswith('finger_'):continue
        rank=int(col.name.split('_')[-1])-1
        mat=bpy.data.materials[palette[f['identities'][rank]]]
        for obj in col.objects:
            if obj.type!='MESH':continue
            for slot in obj.material_slots:
                m=slot.material
                if m and m.name.startswith('f') and m.name[1:].isdigit():slot.material=mat
    link_data={}
    if 'finger_tip_scales' in f:
        from render_fingertip_geometry import measure_tips
        link_data.update(measure_tips(native))
    if 'palm_surface' in f:
        from render_contact_surfaces import surface_mesh_stats
        link_data.update(surface_meshes=surface_mesh_stats())
    if 'finger_codes' in f:
        link_data.update(native_finger_codes=[d.cfg.code for d in native.fingers],native_thumb_code=native.thumbs[0].cfg.code,
                         native_joint_types=[[int(e.id) for e in d.elements if e.type=='joint'] for d in native.fingers+native.thumbs])
    if 'finger_link_lengths_mm' in f:
        digits=native.fingers+native.thumbs
        link_data=dict(native_link_lengths_mm=[[float(e.link_added_length_mm) for e in d.elements
                      if hasattr(e,'link_added_length_mm')] for d in digits],
                      native_tip_positions_mm=[d.elements[-1].transformation[:3,3].tolist() for d in digits])
    return dict(**link_data,thumb_base_mm=cfg.thumb_bases.tolist()[0],thumb_normal=cfg.thumb_bases_normal_vectors.tolist()[0],
                bases_mm=cfg.finger_bases.tolist(),normals=cfg.finger_bases_normal_vectors.tolist(),
                finger_count=n,thumb_count=1,thumb_code=thumb_code,finger_code=codes[0] if len(set(codes))==1 else 'mixed')

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--plan',action='store_true');parser.add_argument('--indices',nargs='+',type=int)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT=args.output.resolve()
    if not OUT.is_relative_to(ROOT):raise ValueError('Output must remain inside website_assets.')
    OUT.mkdir(parents=True,exist_ok=True)
    if args.plan:plan();return
    schedule=json.loads((OUT/'plan.json').read_text())
    for name,digest in schedule['source_hashes'].items():assert hashlib.sha256((GEN/name).read_bytes()).hexdigest()==digest
    indices=args.indices or list(range(len(schedule['frames'])))
    temp=OUT/'tmp'/f'batch-{indices[0]:04d}';temp.mkdir(parents=True,exist_ok=True)
    frames=OUT/'frames';frames.mkdir(exist_ok=True)
    tempfile.tempdir=str(temp);os.environ['TEMP']=os.environ['TMP']=str(temp)
    bpy.context.preferences.filepaths.temporary_directory=str(temp);bpy.context.preferences.filepaths.save_version=0
    addon.register();s=bpy.context.scene.handgen_settings;s.auto_update=False
    s.generation_v2_dir=str(GEN);s.save_output_dir=str(temp);s.generate_collision_mesh=False
    s.show_fingers=True;s.show_thumbs=True;s.thumb_number=1
    for index in indices:
        f=schedule['frames'][index];path=frames/f'{index:04d}.png';meta=path.with_suffix('.json')
        if path.exists() and meta.exists() and json.loads(meta.read_text()).get('render_revision')==f.get('render_revision'):continue
        s.finger_number=len(f['locations'])
        start=time.monotonic();geometry=generate_frame(f,temp)
        scene=bpy.context.scene;style(scene);scene.camera.location=(0,0,1);scene.camera.data.ortho_scale=schedule.get('camera_scale',.54)
        if 'camera_target' in schedule:
            from mathutils import Matrix
            target=Vector(schedule['camera_target'])
            scene.camera.location=target+Vector(schedule['camera_offset'])
            forward=(target-scene.camera.location).normalized()
            right=forward.cross(Vector((0,1,0))).normalized()
            up=right.cross(forward).normalized()
            scene.camera.rotation_euler=Matrix((right,up,-forward)).transposed().to_euler()
        bpy.context.view_layer.update()
        from bpy_extras.object_utils import world_to_camera_view
        visible=[o for o in scene.objects if o.type=='MESH' and o.name.lower().startswith('viz_') and not o.hide_render]
        points=[world_to_camera_view(scene,scene.camera,o.matrix_world@Vector(v)) for o in visible for v in o.bound_box]
        bounds=[min(p.x for p in points),min(p.y for p in points),max(p.x for p in points),max(p.y for p in points)]
        assert all(.025<v<.975 for v in bounds),bounds
        scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
        meta.write_text(json.dumps({**f,**geometry,'bounds':bounds},indent=2))
        if index in schedule.get('keyframes',(91,121,152,183)):bpy.ops.wm.save_as_mainfile(filepath=str(OUT/f'stage-{index:04d}.blend'),copy=True)
        print(f'ASSEMBLY_FRAME {index} {time.monotonic()-start:.1f}s',flush=True)

if __name__=='__main__':main()
