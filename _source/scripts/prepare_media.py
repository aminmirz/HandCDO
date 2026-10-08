"""Create lightweight web derivatives without changing any source assets."""
from pathlib import Path
import subprocess
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'site' / 'media'
OUT.mkdir(parents=True, exist_ok=True)

def main():
    thumbs = []
    for source in sorted((ROOT / 'assets').glob('*.mp4')):
        target = OUT / (source.stem + '.webp')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', '1', '-i', str(source), '-frames:v', '1', '-vf', 'scale=1200:-1', str(target)], check=True)
        img = Image.open(target).convert('RGB')
        tile = Image.new('RGB', (480, 310), '#efeee9')
        tile.paste(ImageOps.contain(img, (480, 278)), (0, 0))
        ImageDraw.Draw(tile).text((12, 285), source.name, fill='black')
        thumbs.append(tile)
    for source in sorted((ROOT / 'figs').glob('*.jpg')):
        img = Image.open(source)
        img.thumbnail((2000, 1600))
        img.save(OUT / (source.stem + '.webp'), quality=88)
    sheet = Image.new('RGB', (1440, 930), '#efeee9')
    for i, tile in enumerate(thumbs):
        sheet.paste(tile, ((i % 3) * 480, (i // 3) * 310))
    sheet.save(ROOT / 'preview' / 'asset-contact-sheet.jpg')
    print('Generated web images and contact sheet.')

if __name__ == '__main__':
    main()
