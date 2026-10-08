"""Validate the encoded research animations and changes in their rendered geometry."""
import json
from pathlib import Path
import subprocess
from PIL import Image,ImageChops

ROOT=Path(__file__).resolve().parents[1]

def main():
    media=ROOT/'site/media/parameters'
    catalog=json.loads((media/'catalog.json').read_text())
    assert len(catalog)==10
    reports=[]
    for item in catalog:
        study=item['study'];video=media/f'{study}.mp4'
        info=json.loads((media/f'{study}.json').read_text())
        assert info['materials']=='Original generation_v2 materials; no palette substitutions'
        result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_frames,pix_fmt','-of','json',str(video)],check=True,capture_output=True,text=True)
        stream=json.loads(result.stdout)['streams'][0]
        assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
        assert (stream['width'],stream['height'])==(1280,1040)
        assert stream['r_frame_rate']=='24/1'
        assert int(stream['nb_frames'])==info['video_frames']
        subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True,capture_output=True)
        raw=ROOT/'.runtime/cycles-render/renders'/study
        start=Image.open(raw/'0000.png').convert('RGB')
        end=Image.open(raw/f'{len(info["frames"])-1:04d}.png').convert('RGB')
        difference=ImageChops.difference(start,end)
        assert difference.getbbox() is not None,study+' has no geometric animation'
        # Labels are added later; this compares only Cycles-rendered images.
        assert sum(v for i,v in enumerate(difference.convert('L').histogram()) if i>=8)>500,study
        if study=='palm-surface':
            peak0=info['frames'][0]['pad_local_z_range_mm'][1]
            peak1=info['frames'][-1]['pad_local_z_range_mm'][1]
            assert 8.5<peak1-peak0<9.5,(peak0,peak1)
        reports.append({'study':study,'frames':info['video_frames'],'seconds':info['duration_seconds'],
                        'bytes':video.stat().st_size,'original_materials':True,'geometry_changes':True,'full_decode':True})
    report={'studies':reports,'total_video_bytes':sum(r['bytes'] for r in reports)}
    (ROOT/'preview/parameter-media-checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
