"""Run in Blender: compare exported meshes and joint motion to the saved native model."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import base64
import hashlib
import importlib.util
import json
import bpy
import numpy as np
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT.parent / 'generation_v2'


def main():
    model_id = sys.argv[sys.argv.index('--') + 1] if '--' in sys.argv else 'baseline'
    output = ROOT / 'site/models' / model_id
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError('Invalid model location.')
    data = json.loads((output / 'model.js').read_text()[len('window.HandCDOModels.register('):-3])
    spec = importlib.util.spec_from_file_location('handcdo_verification_addon', GENERATOR / 'blender/HandGeneratorV2.py')
    addon = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = addon
    spec.loader.exec_module(addon)
    addon.register()
    bpy.ops.wm.open_mainfile(filepath=str(output / 'assembled_hand_model.blend'))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    compared = 0
    for node in data['nodes']:
        if 'geometry' not in node:
            continue
        obj = bpy.data.objects[node['name']]
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        actual = np.empty(len(mesh.vertices) * 3, dtype='<f4')
        mesh.vertices.foreach_get('co', actual)
        geometry = data['geometries'][node['geometry']]
        expected = np.frombuffer(base64.b64decode(geometry['positions']), dtype='<f4')
        np.testing.assert_array_equal(actual, expected)
        mesh.calc_loop_triangles()
        indices = np.empty(len(mesh.loop_triangles) * 3, dtype='<i4')
        mesh.loop_triangles.foreach_get('vertices', indices)
        material_indices = np.empty(len(mesh.loop_triangles), dtype='<i4')
        mesh.loop_triangles.foreach_get('material_index', material_indices)
        indices = indices.reshape(-1, 3)[np.argsort(material_indices, kind='stable')].astype('<u4').reshape(-1)
        expected_indices = np.frombuffer(base64.b64decode(geometry['indices']), dtype='<u4')
        np.testing.assert_array_equal(indices, expected_indices)
        evaluated.to_mesh_clear()
        compared += 1
    assert 'palm' in [m['name'] for m in data['materials']]
    for filename, expected_hash in data['source_hashes'].items():
        assert hashlib.sha256((GENERATOR / filename).read_bytes()).hexdigest() == expected_hash, filename

    local = [Matrix(np.array(node['matrix']).reshape(4, 4).T.tolist()) for node in data['nodes']]
    for joint in data['joints']:
        native = bpy.context.scene.handgen_joints.get(joint['name'])
        assert native is not None
        assert abs(native.min_val - joint['lower']) < 1e-7
        assert abs(native.max_val - joint['upper']) < 1e-7
        angle = max(joint['lower'], min(joint['upper'], joint['upper'] * .42))
        native.value = angle
        local[joint['node']] = local[joint['node']] @ Matrix.Rotation(angle, 4, Vector(joint['axis']))
    bpy.context.view_layer.update()
    memo = {}
    def world(index):
        if index not in memo:
            parent = data['nodes'][index]['parent']
            memo[index] = world(parent) @ local[index] if parent >= 0 else local[index]
        return memo[index]
    maximum_error = 0
    for index, node in enumerate(data['nodes']):
        if 'geometry' not in node:
            continue
        actual = np.asarray(bpy.data.objects[node['name']].matrix_world)
        expected = np.asarray(world(index))
        maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, atol=2e-6)
    report = {'model': model_id, 'meshes_compared_bit_for_bit': compared,
              'joints_compared': len(data['joints']), 'maximum_posed_matrix_error': maximum_error,
              'source_hashes_verified': True}
    (ROOT / 'preview' / f'export-verification-{model_id}.json').write_text(json.dumps(report, indent=2))
    print('EXPORT_VERIFICATION ' + json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
