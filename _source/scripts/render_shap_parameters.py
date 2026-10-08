"""Blender side: render one native generation_v2 hand per frame for the SHAP parameter clips.

Same pipeline as the approved animations (render_thumb_assembly.py): native PalmConfig/FingerConfig/Hand,
original components.blend materials, Workbench + mylight.sl + Both cavity, fixed front orthographic camera.
Adds the per-digit side offsets and thumb offsets used by the paper's optimizer.
Run through build_shap_parameters.py.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_thumb_assembly import *  # noqa: F401,F403  (palm_config, style, addon, assembly, Hand, configs, numpy, bpy)

OUT=ROOT/'preview/shap-parameters'

def generate(f,temp):
    cfg,curve=palm_config(f['thumb_location'])
    cfg.thumb_angle_deg_list[0]+=f.get('thumb_angle_deg',0.)
    cfg.thumb_base_normal_offset_mm_list=[f.get('thumb_normal_offset_mm',0.)]
    cfg.thumb_base_side_offset_mm_list=[f.get('thumb_side_offset_mm',0.)]
    n=len(f['locations']);cfg.finger_number=n
    cfg.finger_location_list=f['locations']
    # Paper semantics (iros_paper_codebase PalmClass): side offset is a straight translation along the
    # boundary tangent t_hat, positive toward +x on the top edge. The approved pipeline places finger points
    # directly, which bypasses v2's own arc-length side offset, so the translation is applied here.
    def placed(t,side):
        p=np.asarray(cfg._curve_point_at(curve,t),dtype=float)
        if not side:return p
        d=np.asarray(cfg._curve_point_at(curve,min(t+1e-3,1.)),dtype=float)-np.asarray(cfg._curve_point_at(curve,max(t-1e-3,0.)),dtype=float)
        return p-side*d/np.linalg.norm(d)
    cfg.finger_points=np.array([placed(t,s) for t,s in zip(f['locations'],f['side_offsets_mm'])]).reshape(n,2)
    # The paper keeps the finger direction of the original boundary point; v2 takes the normal of the outline
    # point nearest the shifted point (PalmClass.finger_point_normal_vector). Add the difference to the angle.
    outline=np.asarray(cfg.outline_points,dtype=float)
    def normal_at(p):
        i=int(np.argmin(np.linalg.norm(outline-p,axis=1)));t=outline[(i+1)%len(outline)]-outline[i-1]
        t=t/np.linalg.norm(t);return np.array([t[1],-t[0]])
    heading=lambda v:math.degrees(math.atan2(v[1],v[0]))
    correction=[heading(normal_at(p))-heading(normal_at(np.asarray(cfg._curve_point_at(curve,t),dtype=float))) if s else 0.
                for p,t,s in zip(cfg.finger_points,f['locations'],f['side_offsets_mm'])]
    cfg.finger_angle_deg_list=[a+c for a,c in zip(f['angles_deg'],correction)]
    cfg.finger_base_normal_offset_mm_list=list(f['normal_offsets_mm'])
    cfg.finger_base_side_offset_mm_list=list(f['side_offsets_mm'])
    fingers=[FingerConfig(type='finger',data='fixed',code=f['finger_codes'][i],id=i) for i in range(n)]
    thumb=FingerConfig(type='thumb',data='fixed',code=f['thumb_code'],id=0)
    for digit,scale in zip(fingers,f['finger_tip_scales']):digit.fingertip_scale_factor=tuple(scale)
    thumb.fingertip_scale_factor=tuple(f['thumb_tip_scale'])
    for digit,length in zip(fingers,f['finger_link_mm']):digit.link_added_length_mm_list=[float(length)]*len(digit.link_added_length_mm_list)
    thumb.link_added_length_mm_list=[float(f['thumb_link_mm'])]*len(thumb.link_added_length_mm_list)
    if f.get('palm_surface'):
        # palm kernels only (the paper's K0/K1); finger pads keep the native no-bump defaults
        from render_contact_surfaces import FIELDS
        p=f['palm_surface'];cfg.pad_resolution_level=6;cfg.bump_type='gaussian'
        cfg.bumps_number=len(p['spread']);cfg.bump_max_height_intensity_mm=p['height_mm']
        for key,field in FIELDS.items():setattr(cfg,field,list(p[key]))
    hc=HandConfig();hc.collision_mesh=False
    native=Hand(cfg,fingers,[thumb],hc,root_dir=str(temp))
    data=native.save_assembly_data();palm=native.save_palm_data()
    assert data and palm and len(native.fingers)==n and len(native.thumbs)==1
    addon._clear_generated_hand();addon._clear_template_objects()
    appended=addon._append_all_from_blend(str(GEN/'blender/components.blend'))
    assembly.FingerAssembly.from_existing_scene(data,appended).run_assembly()
    assembly.PalmMesh(palm).generate();assembly.assign_viz_materials();assembly.apply_origin_offset(palm)
    addon._clear_template_objects();assembly.shade_all_meshes_flat();addon._apply_hand_visibility(bpy.context)
    # Keep each finger's approved colour (F0 green, F1 yellow, F2 blue) whatever its native index.
    palette=['t','f3','f2','f1']
    missing=[name for name in palette if bpy.data.materials.get(name) is None]
    if missing:
        with bpy.data.libraries.load(str(GEN/'blender/components.blend'),link=False) as (src,dst):dst.materials=missing
    for col in bpy.data.collections['hand'].children:
        if not col.name.startswith('finger_'):continue
        mat=bpy.data.materials[palette[f['identities'][int(col.name.split('_')[-1])-1]]]
        for obj in col.objects:
            if obj.type!='MESH':continue
            for slot in obj.material_slots:
                if slot.material and slot.material.name.startswith('f') and slot.material.name[1:].isdigit():slot.material=mat
    return dict(bases_mm=cfg.finger_bases.tolist(),normals=cfg.finger_bases_normal_vectors.tolist(),thumb_base_mm=cfg.thumb_bases.tolist()[0],
                native_finger_codes=[d.cfg.code for d in native.fingers],native_thumb_code=native.thumbs[0].cfg.code)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--indices',nargs='+',type=int,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    schedule=json.loads((OUT/'plan.json').read_text())
    for name,digest in schedule['source_hashes'].items():assert hashlib.sha256((GEN/name).read_bytes()).hexdigest()==digest
    temp=OUT/'tmp'/f'batch-{args.indices[0]:04d}';temp.mkdir(parents=True,exist_ok=True)
    frames=OUT/'frames';frames.mkdir(exist_ok=True)
    tempfile.tempdir=str(temp);os.environ['TEMP']=os.environ['TMP']=str(temp)
    bpy.context.preferences.filepaths.temporary_directory=str(temp);bpy.context.preferences.filepaths.save_version=0
    addon.register();s=bpy.context.scene.handgen_settings;s.auto_update=False
    s.generation_v2_dir=str(GEN);s.save_output_dir=str(temp);s.generate_collision_mesh=False
    s.show_fingers=True;s.show_thumbs=True;s.thumb_number=1
    from bpy_extras.object_utils import world_to_camera_view
    for index in args.indices:
        f=schedule['frames'][index];path=frames/f"{f['name']}.png";meta=path.with_suffix('.json')
        if path.exists() and meta.exists():continue
        s.finger_number=len(f['locations']);start=time.monotonic()
        geometry=generate(f,temp)
        scene=bpy.context.scene;style(scene)
        scene.camera.location=(0,0,1);scene.camera.data.ortho_scale=schedule['camera_scale']
        bpy.context.view_layer.update()
        visible=[o for o in scene.objects if o.type=='MESH' and o.name.lower().startswith('viz_') and not o.hide_render]
        points=[world_to_camera_view(scene,scene.camera,o.matrix_world@Vector(v)) for o in visible for v in o.bound_box]
        bounds=[min(p.x for p in points),min(p.y for p in points),max(p.x for p in points),max(p.y for p in points)]
        assert all(.025<v<.975 for v in bounds),bounds
        scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
        meta.write_text(json.dumps({**f,**geometry,'bounds':bounds,'engine':'BLENDER_WORKBENCH'},indent=2))
        print(f"SHAP_FRAME {f['name']} {time.monotonic()-start:.1f}s",flush=True)

if __name__=='__main__':main()
