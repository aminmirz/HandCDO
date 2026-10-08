"""Encode genuine Cycles frames as labeled, white-background website videos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]

def encode(source, target):
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['engine']=='CYCLES' and manifest['device']=='OPTIX'
    assert manifest.get('materials')=='Original generation_v2 materials; no palette substitutions', 'Draft material set cannot be published'
    for relative,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT.parent/'generation_v2'/relative).read_bytes()).hexdigest()==digest, relative
    frames=manifest['frames']
    assert [f['index'] for f in frames]==list(range(len(frames)))
    assert len(frames) in (4,49)
    for f in frames:
        assert all(0<=v<=1 for v in f['bounds']), (manifest['study'],f['index'],f['bounds'])
    width,height=manifest['width'],manifest['height']
    footer=80
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',44)
    rendered=[]
    for frame in frames:
        img=Image.open(source/f'{frame["index"]:04d}.png').convert('RGB')
        assert img.size==(width,height)
        # OptiX can leave slight color noise at the image boundary. Restore the
        # four-pixel perimeter to white, strictly outside the projected model.
        x0,y0,x1,y1=frame['bounds']
        assert min(x0*width,y0*height,(1-x1)*width,(1-y1)*height)>4
        for xy in [(0,0),(width-1,0),(0,height-1),(width-1,height-1)]:
            assert min(img.getpixel(xy))>=238, ('Non-white background',source,xy)
        canvas=Image.new('RGB',(width,height+footer),'white')
        canvas.paste(img.crop((4,4,width-4,height-4)),(4,footer+4))
        draw=ImageDraw.Draw(canvas)
        draw.line((40,footer-4,width-40,footer-4),fill='#e3e7eb',width=1)
        assert draw.textbbox((40,16),frame['label'],font=font)[2]<width-30
        draw.text((40,16),frame['label'],font=font,fill='#13294b')
        rendered.append(canvas)
    if len(frames)==4:
        order=[i for i in [0,1,2,3,2,1] for _ in range(36)]
    else:
        order=[0]*12+list(range(49))+[48]*12+list(range(47,0,-1))
    target.mkdir(parents=True,exist_ok=True)
    video=target/(manifest['study']+'.mp4')
    cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height+footer}',
         '-r','24','-i','-','-an','-c:v','libx264','-preset','slow','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i in order:process.stdin.write(rendered[i].tobytes())
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('FFmpeg failed')
    finally:
        if process.poll() is None:process.kill()
    poster=target/(manifest['study']+'.webp')
    rendered[len(frames)//2].save(poster,quality=90)
    manifest.update(fps=24,video_frames=len(order),duration_seconds=len(order)/24,
                    video_width=width,video_height=height+footer,
                    border_cleanup='Four-pixel white perimeter outside projected model bounds',
                    sequence='Discrete configurations with direct cuts' if len(frames)==4 else 'Eased forward/reverse parameter sweep with endpoint holds',
                    file_bytes=video.stat().st_size)
    (target/(manifest['study']+'.json')).write_text(json.dumps(manifest,indent=2))
    print(manifest['study'],len(order),'frames',video.stat().st_size,'bytes',flush=True)
    return {k:manifest[k] for k in ('study','title','description','engine','device','gpu','fps','video_frames','duration_seconds','video_width','video_height','file_bytes')}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'.runtime/cycles-render/renders')
    parser.add_argument('--studies',nargs='*')
    args=parser.parse_args()
    target=ROOT/'site/media/parameters'
    reports=[]
    for source in sorted(args.source.iterdir()):
        if source.is_dir() and (source/'manifest.json').exists() and (not args.studies or source.name in args.studies):
            reports.append(encode(source,target))
    if not args.studies:(target/'catalog.json').write_text(json.dumps(reports,indent=2))

if __name__=='__main__':main()
