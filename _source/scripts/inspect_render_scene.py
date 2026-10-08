import sys
sys.dont_write_bytecode=True
import bpy, json
from mathutils import Vector
print('BLENDER',bpy.app.version_string)
p=bpy.context.preferences.addons['cycles'].preferences
try:
 p.compute_device_type='OPTIX'; p.get_devices()
 print('DEVICES',[(d.name,d.type) for d in p.devices])
except Exception as e: print('DEVICE_ERROR',str(e))
for mod in ['numpy','scipy','pyclipper']:
 try:
  m=__import__(mod); print('DEPENDENCY',mod,getattr(m,'__version__','ok'))
 except ImportError: print('MISSING',mod)
for mat in bpy.data.materials:
 n=next((n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None) if mat.use_nodes else None
 print('MATERIAL',mat.name, list(n.inputs['Base Color'].default_value) if n else list(mat.diffuse_color),n.inputs['Roughness'].default_value if n else '')
for obj in bpy.context.scene.objects:
 if obj.type in ['CAMERA','LIGHT']:
  print('STUDIO',obj.name,list(obj.location),list(obj.rotation_euler))
pts=[o.matrix_world@Vector(p) for o in bpy.context.scene.objects if o.type=='MESH' and not o.hide_render and o.name.lower().startswith('viz_') for p in o.bound_box]
if pts: print('BOUNDS',[(min(p[i] for p in pts),max(p[i] for p in pts)) for i in range(3)])
