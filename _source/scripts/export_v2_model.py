"""Run inside Blender to export the unmodified generation_v2 add-on output.

blender --background --factory-startup --python-exit-code 1 --python export_v2_model.py -- --id baseline
All outputs, Python caches, and the add-on's temporary assembly data stay in website_assets.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse
import base64
import hashlib
import importlib.util
import json
import os
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT.parent / 'generation_v2'
DEPENDENCIES = ROOT / '.runtime' / 'python'
if DEPENDENCIES.exists():
    sys.path.insert(0, str(DEPENDENCIES))


def export_scene(output, model_id, parameters):
    import bpy
    import numpy as np
    hand = bpy.data.collections.get('hand')
    if hand is None:
        raise RuntimeError('Generator did not create a hand collection.')
    bpy.context.view_layer.update()
    # Palm meshes are linked outside the finger collection. Templates have been
    # removed by the add-on, so collect all generated viz_ meshes in the scene.
    visible = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH' and obj.name.lower().startswith('viz_') and not obj.hide_render]
    if not visible:
        raise RuntimeError('Generator did not create visible hand meshes.')
    if not any('palm' in obj.name.lower() for obj in visible):
        raise RuntimeError('Export selection is missing the generated palm.')
    included = set(visible)
    for obj in visible:
        ancestor = obj.parent
        while ancestor is not None:
            included.add(ancestor)
            ancestor = ancestor.parent
    objects = sorted(included, key=lambda obj: obj.name)
    ids = {obj: index for index, obj in enumerate(objects)}
    materials, material_ids, geometries, geometry_ids, nodes = [], {}, [], {}, []
    depsgraph = bpy.context.evaluated_depsgraph_get()

    def encode(array):
        return base64.b64encode(array.tobytes()).decode('ascii')

    def material_id(material):
        name = material.name if material else 'default'
        if name not in material_ids:
            color = list(material.diffuse_color) if material else [.35, .35, .35, 1]
            roughness, metalness = .65, 0
            if material and material.use_nodes:
                bsdf = next((n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
                if bsdf:
                    color = list(bsdf.inputs['Base Color'].default_value)
                    roughness = float(bsdf.inputs['Roughness'].default_value)
                    metalness = float(bsdf.inputs['Metallic'].default_value)
            material_ids[name] = len(materials)
            materials.append({'name': name, 'color': color[:3], 'opacity': color[3], 'roughness': roughness, 'metalness': metalness})
        return material_ids[name]

    for obj in objects:
        local = obj.parent.matrix_world.inverted_safe() @ obj.matrix_world if obj.parent in included else obj.matrix_world
        node = {'name': obj.name, 'parent': ids.get(obj.parent, -1), 'matrix': [float(local[row][col]) for col in range(4) for row in range(4)]}
        if obj in visible:
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            mesh.calc_loop_triangles()
            positions = np.empty(len(mesh.vertices) * 3, dtype='<f4')
            mesh.vertices.foreach_get('co', positions)
            triangles = np.empty(len(mesh.loop_triangles) * 3, dtype='<i4')
            mesh.loop_triangles.foreach_get('vertices', triangles)
            mat_indices = np.empty(len(mesh.loop_triangles), dtype='<i4')
            mesh.loop_triangles.foreach_get('material_index', mat_indices)
            order = np.argsort(mat_indices, kind='stable')
            indices = triangles.reshape(-1, 3)[order].astype('<u4').reshape(-1)
            sorted_materials = mat_indices[order]
            groups = []
            for slot in np.unique(sorted_materials):
                matches = np.flatnonzero(sorted_materials == slot)
                groups.append([int(matches[0]) * 3, len(matches) * 3, int(slot)])
            signature = hashlib.sha256(positions.tobytes() + indices.tobytes() + json.dumps(groups).encode()).hexdigest()
            if signature not in geometry_ids:
                geometry_ids[signature] = len(geometries)
                geometries.append({'positions': encode(positions), 'indices': encode(indices), 'groups': groups, 'vertices': len(mesh.vertices), 'triangles': len(mesh.loop_triangles)})
            node['geometry'] = geometry_ids[signature]
            node['materials'] = [material_id(slot.material) for slot in obj.material_slots] or [material_id(None)]
            evaluated.to_mesh_clear()
        nodes.append(node)

    joints = []
    for joint in bpy.context.scene.handgen_joints:
        obj = bpy.data.objects.get(joint.link_frame_name)
        if obj in ids:
            joints.append({'name': joint.name, 'node': ids[obj], 'axis': [joint.axis_x, joint.axis_y, joint.axis_z], 'lower': joint.min_val, 'upper': joint.max_val, 'chain': joint.chain_id})
    hashes = {str(path.relative_to(GENERATOR)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(GENERATOR.rglob('*')) if path.suffix in ('.py', '.blend')}
    model = {'schema': 'handcdo-generation-v2-mesh/1', 'id': model_id, 'parameters': parameters,
             'source': 'generation_v2/blender/HandGeneratorV2.py', 'source_hashes': hashes,
             'blender': bpy.app.version_string, 'units': 'meters', 'materials': materials,
             'geometries': geometries, 'nodes': nodes, 'joints': joints,
             'stats': {'meshes': len(visible), 'unique_geometries': len(geometries),
                       'triangles': sum(geometries[node['geometry']]['triangles'] for node in nodes if 'geometry' in node), 'joints': len(joints)}}
    (output / 'model.js').write_text('window.HandCDOModels.register(' + json.dumps(model, separators=(',', ':'), allow_nan=False) + ');\n', encoding='utf-8')
    # Plain metadata and native config files provide reproducible provenance.
    metadata = {key: value for key, value in model.items() if key not in ('geometries', 'nodes', 'materials')}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    print('WEBSITE_EXPORT ' + json.dumps(model['stats']), flush=True)


def main():
    import bpy
    import pyclipper
    from scipy.spatial import cKDTree
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='baseline')
    parser.add_argument('--parameters', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--save-blend', action='store_true')
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    args = parser.parse_args(argv)
    if not args.id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.id):
        raise ValueError('Invalid model identifier.')
    output = (args.output or ROOT / 'site' / 'models' / args.id).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must remain under website_assets.')
    output.mkdir(parents=True, exist_ok=True)
    temp = output / 'generation'
    temp.mkdir(exist_ok=True)
    tempfile.tempdir = str(temp)
    os.environ['TEMP'] = os.environ['TMP'] = str(temp)
    bpy.context.preferences.filepaths.temporary_directory = str(temp)
    parameters = json.loads(args.parameters.read_text()) if args.parameters else {}
    start = time.monotonic()
    spec = importlib.util.spec_from_file_location('handcdo_v2_addon', GENERATOR / 'blender' / 'HandGeneratorV2.py')
    addon = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = addon
    spec.loader.exec_module(addon)
    addon.register()
    settings = bpy.context.scene.handgen_settings
    settings.auto_update = False
    settings.generation_v2_dir = str(GENERATOR)
    settings.save_output_dir = str(output)
    settings.generate_collision_mesh = False
    for key, value in parameters.items():
        if key not in settings.bl_rna.properties or key in ('rna_type', 'auto_update', 'generation_v2_dir', 'save_output_dir'):
            raise ValueError('Unknown or reserved generator parameter: ' + key)
        setattr(settings, key, value)
    result = addon.generate_hand(bpy.context)
    if result != {'FINISHED'}:
        raise RuntimeError('Hand generation failed: ' + str(result))
    palm_data = bpy.context.scene.get('handgen_last_palm_npz', '')
    if not palm_data or not Path(palm_data).is_file() or bpy.data.objects.get('PalmBody_root') is None:
        raise RuntimeError('Generator did not produce a complete palm; see generation log.')
    actual = {}
    for prop in settings.bl_rna.properties:
        if prop.identifier in ('rna_type', 'generation_v2_dir', 'save_output_dir', 'save_hand_name', 'auto_update'):
            continue
        value = getattr(settings, prop.identifier)
        actual[prop.identifier] = list(value) if getattr(prop, 'is_array', False) else value
    (output / 'parameters.json').write_text(json.dumps(actual, indent=2), encoding='utf-8')
    export_scene(output, args.id, actual)
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(output / 'assembled_hand_model.blend'), copy=True)
    print('WEBSITE_EXPORT_SECONDS ' + str(round(time.monotonic() - start, 2)), flush=True)


if __name__ == '__main__':
    main()
