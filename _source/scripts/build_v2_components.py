"""Export the individual figure components from the paper / presentation PowerPoint files for the v2 page.

Each component is the original embedded image, cut with the same crop rectangle (a:srcRect) the slide uses,
so the web page shows the source renders and photos rather than crops of the flattened figures.

  python -B website_assets/scripts/build_v2_components.py [--figures PATH] [--presentation PATH]
"""
import argparse,io,json,re,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'v2/media/components'
HCD=Path.home()/'Desktop/PhD/_Projects/HandCoDesign'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
REL='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed'

# (output name, deck, slide number, shape name) ; shape names are the PowerPoint object names
COMPONENTS=[
    # Teaser (figures.pptx slide 4)
    ('teaser-sim-before','fig',4,'Google Shape;235;p16'),('teaser-sim-after','fig',4,'Google Shape;236;p16'),
    ('teaser-real-a','fig',4,'Google Shape;240;p16'),('teaser-real-b','fig',4,'Google Shape;238;p16'),
    ('teaser-unstable','fig',4,'Google Shape;239;p16'),('teaser-stable','fig',4,'Google Shape;237;p16'),
    # Framework (slide 3)
    ('fw-hand-1','fig',3,'Google Shape;130;p15'),('fw-hand-2','fig',3,'Google Shape;129;p15'),('fw-hand-3','fig',3,'Google Shape;128;p15'),
    ('fw-demo','fig',3,'Google Shape;225;p15'),('fw-grasp-box','fig',3,'Google Shape;111;p15'),
    ('fw-joint-sampling','fig',3,'Google Shape;108;p15'),('fw-wrench','fig',3,'Google Shape;106;p15'),
    ('fw-tool-a','fig',3,'Google Shape;224;p15'),('fw-tool-b','fig',3,'Google Shape;229;p15'),
    # Results (slide 7)
    ('res-progress','fig',7,'Google Shape;460;p19'),('res-group-shap','fig',7,'Google Shape;459;p19'),('res-shap','fig',7,'Google Shape;461;p19'),
    ('res-sim-high','fig',7,'Google Shape;442;p19'),('res-sim-mid','fig',7,'Google Shape;444;p19'),('res-sim-low','fig',7,'Google Shape;443;p19'),
    ('res-fab-high','fig',7,'Google Shape;452;p19'),('res-fab-mid','fig',7,'Google Shape;451;p19'),('res-fab-low','fig',7,'Google Shape;453;p19'),
    # Experiments (slide 2)
    ('exp-slip','fig',2,'Google Shape;100;p14'),('exp-cut','fig',2,'Google Shape;97;p14'),
    ('exp-hammer','fig',2,'Google Shape;96;p14'),('exp-stir','fig',2,'Google Shape;95;p14'),
    # Palm generation (slide 1)
    ('palm-init','fig',1,'Google Shape;54;p13'),('palm-outline-a','fig',1,'Google Shape;65;p13'),('palm-outline-b','fig',1,'Google Shape;66;p13'),
    ('palm-mesh-a','fig',1,'Google Shape;59;p13'),('palm-mesh-b','fig',1,'Google Shape;58;p13'),
    # Finger generation (slide 5)
    ('finger-code','fig',5,'Google Shape;278;p17'),('finger-hand','fig',5,'Google Shape;361;p17'),('thumb-hand','fig',5,'Google Shape;280;p17'),
    ('thumb-code','fig',5,'Google Shape;281;p17'),('finger-tip','fig',5,'Google Shape;283;p17'),
    ('mode-0','fig',5,'Google Shape;381;p17'),('mode-1','fig',5,'Google Shape;374;p17'),('mode-2','fig',5,'Google Shape;370;p17'),('mode-3','fig',5,'Google Shape;377;p17'),
    ('joint-short','fig',5,'Google Shape;385;p17'),('joint-long','fig',5,'Google Shape;388;p17'),('link-spacer','fig',5,'Google Shape;391;p17'),
    ('thumb-mode-0','fig',5,'Google Shape;363;p17'),('thumb-mode-1','fig',5,'Google Shape;366;p17'),
    # Surface kernels (slide 6)
    ('kernel-3d','fig',6,'Google Shape;419;p18'),('kernel-heightmap','fig',6,'Google Shape;427;p18'),
    ('kernel-pad','fig',6,'Google Shape;432;p18'),('kernel-colliders','fig',6,'Google Shape;433;p18'),
]

def shapes(z,slide):
    xml=f'ppt/slides/slide{slide}.xml'
    rels={r.get('Id'):r.get('Target') for r in ET.fromstring(z.read(f'ppt/slides/_rels/slide{slide}.xml.rels'))}
    out={}
    for pic in ET.fromstring(z.read(xml)).iter('{%s}pic'%NS['p']):
        name=pic.find('.//p:cNvPr',NS).get('name');blip=pic.find('.//a:blip',NS);src=pic.find('.//a:srcRect',NS)
        crop={k:int(v)/100000 for k,v in (src.attrib.items() if src is not None else [])}
        out[name]=('ppt/'+rels[blip.get(REL)].replace('../',''),crop)
    return out

def cut(img,crop):
    w,h=img.size
    l,t,r,b=(crop.get(k,0) for k in 'ltrb')
    box=[round(l*w),round(t*h),round(w-r*w),round(h-b*h)]
    # negative crop values pad the picture
    pad=[max(0,-box[0]),max(0,-box[1]),max(0,box[2]-w),max(0,box[3]-h)]
    if any(pad):
        canvas=Image.new(img.mode,(w+pad[0]+pad[2],h+pad[1]+pad[3]),(255,255,255,0) if img.mode=='RGBA' else 'white')
        canvas.paste(img,(pad[0],pad[1]));img=canvas;box=[box[0]+pad[0],box[1]+pad[1],box[2]+pad[0],box[3]+pad[1]]
    return img.crop(box)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--figures',type=Path,default=HCD/'figures.pptx')
    args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    decks={'fig':zipfile.ZipFile(args.figures)};cache={};records=[]
    for name,deck,slide,shape in COMPONENTS:
        z=decks[deck]
        if (deck,slide) not in cache:cache[deck,slide]=shapes(z,slide)
        media,crop=cache[deck,slide][shape]
        img=Image.open(io.BytesIO(z.read(media)));img=img.convert('RGBA' if img.mode in ('RGBA','LA','P') else 'RGB')
        img=cut(img,crop);img.thumbnail((1400,1400),Image.Resampling.LANCZOS)
        img.save(OUT/f'{name}.webp',quality=88,method=6)
        records.append(dict(id=name,deck=str(args.figures.name),slide=slide,shape=shape,media=media,crop=crop,size=img.size))
    (OUT/'manifest.json').write_text(json.dumps(records,indent=2))
    print(f'Wrote {len(records)} components to {OUT}')

if __name__=='__main__':main()
