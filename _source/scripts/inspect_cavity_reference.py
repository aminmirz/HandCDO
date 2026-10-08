import sys
sys.dont_write_bytecode=True
import bpy,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
media=root.parent.parent.parent.parent.parent/'media'
media=Path(r'C:/Users/aminm/Desktop/PhD/_Projects/HandCoDesign/media')
names=['type','light','studio_light','color_type','show_shadows','show_cavity','cavity_type','cavity_ridge_factor','cavity_valley_factor','curvature_ridge_factor','curvature_valley_factor','show_object_outline','show_specular_highlight','background_type','background_color','studiolight_rotate_z','studiolight_intensity']
def read(s):
 d={}
 for n in names:
  if hasattr(s,n):
   v=getattr(s,n);d[n]=list(v) if hasattr(v,'__len__') and not isinstance(v,str) else v
 return d
reports=[]
for rel in ['teaser_handgen/hand anime.blend','fingerGen_figure/fingers.blend','Media_1/rendering.blend']:
 bpy.ops.wm.open_mainfile(filepath=str(media/rel),load_ui=True)
 report={'file':rel,'engine':bpy.context.scene.render.engine,'render':read(bpy.context.scene.display.shading),'viewports':[]}
 for screen in bpy.data.screens:
  for area in screen.areas:
   if area.type=='VIEW_3D':report['viewports'].append({'screen':screen.name,**read(area.spaces.active.shading)})
 reports.append(report)
for n in ['cavity_ridge_factor','cavity_valley_factor','curvature_ridge_factor','curvature_valley_factor']:
 p=bpy.context.scene.display.shading.bl_rna.properties[n];print('LIMIT',n,p.hard_max)
(root/'preview/front-cavity/reference-settings.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
