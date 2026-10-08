"""Exercise the local generator through the browser, including a real Blender build."""
import sys
sys.dont_write_bytecode = True
import json
import os
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright, expect
from serve_generator import GeneratorServer
from build_v2_library import find_blender
from generator_config import ROOT, DEFAULTS


def main():
    output = ROOT / 'preview'
    temp = output / 'browser-temp'
    temp.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(temp)
    server = GeneratorServer(0, find_blender())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    errors, checks = [], []
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(executable_path=os.environ.get('HANDCDO_CHROME', r'C:\Program Files\Google\Chrome\Application\chrome.exe'), headless=True, downloads_path=str(output))
            page = browser.new_page(viewport={'width': 1440, 'height': 1100})
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(server.origin, wait_until='networkidle')
            expect(page.locator('#hand-viewer')).to_have_attribute('data-model-id', 'baseline', timeout=30000)
            expect(page.locator('#palm')).to_be_enabled()
            expect(page.locator('#generate-hand')).to_be_enabled()
            checks.append('Local service enables original-generator parameter controls')
            # No token or cross-origin requests can trigger Blender.
            denied = page.request.post(server.origin + '/api/generate', data=DEFAULTS)
            assert denied.status == 403
            headers = {'Origin': server.origin, 'X-HandCDO-Token': server.token}
            bad = page.request.post(server.origin + '/api/generate', headers=headers, data=dict(DEFAULTS, output='../escape'))
            assert bad.status == 400
            bad = page.request.post(server.origin + '/api/generate', headers=headers, data=dict(DEFAULTS, fingers=99))
            assert bad.status == 400
            checks.append('Missing authorization, unknown fields, and out-of-range inputs rejected')
            page.locator('#palm').fill('178')
            page.locator('#length').fill('3')
            page.locator('#surface').fill('2')
            assert page.locator('#hand-viewer').get_attribute('data-model-id') == 'baseline'
            assert 'pending' in page.locator('#generation-status').inner_text()
            checks.append('Pending parameters do not substitute an approximate hand')
            page.locator('#generate-hand').click()
            expect(page.locator('#generation-status')).to_contain_text('Generated with', timeout=210000)
            model_id = page.locator('#hand-viewer').get_attribute('data-model-id')
            assert model_id and model_id != 'baseline'
            metadata = json.loads((ROOT / '.runtime/models' / model_id / 'metadata.json').read_text())
            assert metadata['parameters']['palm_size_mm'] == 178
            assert metadata['parameters']['finger_0_link_added_length'] == 3
            assert metadata['parameters']['pad_max_intensity'] == 2
            assert metadata['stats']['joints'] == 16
            checks.append('Real Blender generation updates the displayed meshes and original parameters')
            page.locator('.explorer-shell').screenshot(path=str(output / 'local-v2-generator.png'))
            with page.expect_download() as download:
                page.locator('#export-design').click()
            download.value.save_as(output / 'tested-live-v2-settings.json')
            exported = json.loads((output / 'tested-live-v2-settings.json').read_text())
            assert exported['palm_size_mm'] == 178
            assert exported['finger_0_link_added_length'] == 3
            checks.append('Live model settings export matches the generated configuration')
            cached = page.request.post(server.origin + '/api/generate', headers=headers, data=dict(DEFAULTS, palm=178, length=3, surface=2))
            assert cached.status == 200 and cached.json()['state'] == 'complete'
            checks.append('Repeated configurations reuse a source-versioned generated model')
            browser.close()
        assert not errors, errors
        report = {'checks': checks, 'javascript_errors': errors, 'model_id': model_id}
        (output / 'generator-checks.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
