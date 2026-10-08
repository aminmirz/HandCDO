"""Generate the static website model library using Blender and generation_v2."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
from generator_config import ROOT, PRESETS, addon_parameters


def find_blender(override=None):
    if override:
        path = Path(override)
    elif os.environ.get('HANDCDO_BLENDER'):
        path = Path(os.environ['HANDCDO_BLENDER'])
    elif shutil.which('blender'):
        path = Path(shutil.which('blender'))
    else:
        choices = sorted(Path('C:/Program Files/Blender Foundation').glob('Blender */blender.exe'))
        if not choices:
            raise FileNotFoundError('Set HANDCDO_BLENDER to a Blender executable (4.5 or later).')
        path = choices[-1]
    if not path.is_file():
        raise FileNotFoundError('Blender executable not found: ' + str(path))
    return str(path.resolve())


def generate(blender, model_id, values, output, save_blend=False):
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must be inside website_assets.')
    output.mkdir(parents=True, exist_ok=True)
    settings = output / 'requested-settings.json'
    settings.write_text(json.dumps(addon_parameters(values), indent=2), encoding='utf-8')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    runtime = ROOT / '.runtime' / 'blender-temp'
    runtime.mkdir(parents=True, exist_ok=True)
    env['TEMP'] = env['TMP'] = str(runtime)
    command = [blender, '--background', '--factory-startup', '--python-exit-code', '1',
               '--python', str(ROOT / 'scripts/export_v2_model.py'), '--', '--id', model_id,
               '--parameters', str(settings), '--output', str(output)]
    if save_blend:
        command.append('--save-blend')
    with (output / 'generation.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    if result.returncode != 0 or not (output / 'model.js').is_file():
        log = (output / 'generation.log').read_text(encoding='utf-8', errors='replace')
        raise RuntimeError('Blender generation failed (exit ' + str(result.returncode) + ').\n' + log[-3500:])
    metadata = json.loads((output / 'metadata.json').read_text())
    metadata['controls'] = values
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output / 'controls.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
    with zipfile.ZipFile(output / 'configuration.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((output / 'generation/handgen_blender').glob('*')):
            if path.suffix in ('.py', '.npz'):
                archive.write(path, path.name)
        for name in ('parameters.json', 'requested-settings.json', 'metadata.json', 'controls.json'):
            archive.write(output / name, name)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blender')
    parser.add_argument('--presets', nargs='+', choices=list(PRESETS), default=list(PRESETS))
    parser.add_argument('--save-blend', action='store_true')
    args = parser.parse_args()
    blender = find_blender(args.blender)
    for model_id in args.presets:
        label, values = PRESETS[model_id]
        print('Generating ' + model_id, flush=True)
        metadata = generate(blender, model_id, values, ROOT / 'site/models' / model_id, args.save_blend)
        print(model_id + ': ' + json.dumps(metadata['stats']), flush=True)
    manifest = []
    for model_id, (label, values) in PRESETS.items():
        if (ROOT / 'site/models' / model_id / 'model.js').exists():
            manifest.append(dict(id=model_id, label=label, controls=values, base=f'site/models/{model_id}'))
    (ROOT / 'site/models/catalog.js').write_text('window.HandCDOCatalog = ' + json.dumps(manifest) + ';\n', encoding='utf-8')


if __name__ == '__main__':
    main()
