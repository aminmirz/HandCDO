"""Check the academic page and its rendered parameter animations in Chrome."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview'
(OUT/'browser-temp').mkdir(parents=True,exist_ok=True)
os.environ['TEMP']=os.environ['TMP']=str(OUT/'browser-temp')

class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self,*args):pass

def main():
    server=ThreadingHTTPServer(('127.0.0.1',0),partial(QuietHandler,directory=str(ROOT)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    origin=f'http://127.0.0.1:{server.server_port}'
    errors,missing,checks=[],[],[]
    try:
        with sync_playwright() as pw:
            browser=pw.chromium.launch(headless=True,executable_path=os.environ.get('HANDCDO_CHROME',r'C:\Program Files\Google\Chrome\Application\chrome.exe'),downloads_path=str(OUT))
            context=browser.new_context(viewport={'width':1440,'height':1000})
            page=context.new_page()
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.on('response',lambda response:missing.append(response.url) if response.status>=400 else None)
            page.goto(origin,wait_until='networkidle')
            assert page.locator('h1').inner_text()=='Function-based Parametric Co-Design Optimization of Dexterous Hands'
            assert page.locator('canvas,#hand-viewer').count()==0
            assert page.locator('script[src]').count()==1
            assert page.locator('.parameter-video').count()==8
            assert page.locator('.parameter-video[src]').count()<=2
            page.screenshot(path=str(OUT/'desktop-hero.png'))
            checks.append('Academic header; model viewer removed; eight videos load on demand')
            seen=[]
            for group in ['palm','placement','kinematics','contact']:
                tab=page.locator(f'#tab-{group}');tab.click()
                expect(tab).to_have_attribute('aria-selected','true')
                assert page.locator('.study-panel:visible').count()==1
                panel=page.locator(f'#studies-{group}')
                for video in panel.locator('video').all():
                    video.scroll_into_view_if_needed()
                    slug=video.get_attribute('data-study')
                    page.wait_for_function('(slug)=>{const v=document.querySelector(`[data-study="${slug}"]`);return v.readyState>=2 && v.videoWidth===800 && v.videoHeight===600;}',arg=slug,timeout=20000)
                    assert video.evaluate('(v)=>v.duration')>=2.9
                    assert video.evaluate('(v)=>v.error===null')
                    seen.append(slug)
                assert page.locator('.study-panel[hidden] video').evaluate_all('(videos)=>videos.every(v=>v.paused)')
                panel.locator('.figure-link').click()
                expect(page.locator('#figure-dialog')).to_be_visible()
                page.keyboard.press('Escape')
                expect(page.locator('#figure-dialog')).not_to_be_visible()
            assert len(set(seen))==8
            checks.append('All eight MP4s decode at 800 x 600; hidden groups pause; figure dialogs work')
            page.locator('#tab-palm').click();page.locator('#tab-palm').focus()
            page.keyboard.press('ArrowRight')
            expect(page.locator('#tab-placement')).to_have_attribute('aria-selected','true')
            page.keyboard.press('End')
            expect(page.locator('#tab-contact')).to_have_attribute('aria-selected','true')
            page.keyboard.press('Home')
            expect(page.locator('#tab-palm')).to_have_attribute('aria-selected','true')
            checks.append('Arrow, Home, and End keyboard navigation across parameter groups')
            page.locator('#study-motion-toggle').click()
            expect(page.locator('#study-motion-toggle')).to_have_attribute('aria-pressed','true')
            assert page.locator('.ambient-video').evaluate_all('(videos)=>videos.every(v=>v.paused)')
            page.locator('#study-motion-toggle').click()
            expect(page.locator('#study-motion-toggle')).to_have_attribute('aria-pressed','false')
            checks.append('Global animation control pauses and resumes playback')
            with page.expect_download() as download:
                page.locator('#studies-palm .study-download').first.click()
            assert download.value.suggested_filename=='palm-geometry.mp4'
            download.value.save_as(OUT/'tested-parameter-animation.mp4')
            checks.append('MP4 download works')
            page.locator('#copy-citation').click()
            expect(page.locator('#copy-citation')).to_contain_text('Copied')
            checks.append('Citation copied to clipboard')
            page.locator('#generation').scroll_into_view_if_needed()
            page.wait_for_timeout(700)
            page.locator('#generation').screenshot(path=str(OUT/'desktop-animations.png'))
            page.screenshot(path=str(OUT/'desktop-full.png'),full_page=True)
            for width in [390,768,1024]:
                page.set_viewport_size({'width':width,'height':844})
                page.wait_for_timeout(100)
                assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),width
                for group in ['palm','placement','kinematics','contact']:
                    page.locator(f'#tab-{group}').click()
                    assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),(width,group)
            page.set_viewport_size({'width':390,'height':844})
            page.evaluate('scrollTo({top:0,behavior:"instant"})')
            page.screenshot(path=str(OUT/'mobile-hero.png'))
            page.locator('#tab-palm').click()
            page.locator('#generation').screenshot(path=str(OUT/'mobile-animations.png'))
            checks.append('All groups fit mobile, tablet, and desktop layouts')
            page.emulate_media(reduced_motion='reduce')
            page.wait_for_timeout(200)
            assert page.locator('.ambient-video').evaluate_all('(videos)=>videos.every(v=>v.paused)')
            assert page.locator('.scroll-reveal').first.evaluate('(e)=>getComputedStyle(e).animationName')=='none'
            checks.append('Reduced motion pauses videos and disables scroll transitions')
            context.close()
            local=browser.new_page(viewport={'width':1280,'height':900})
            local.on('pageerror',lambda error:errors.append(str(error)))
            local.goto((ROOT/'index.html').as_uri(),wait_until='load')
            local.locator('#tab-placement').click()
            local.locator('[data-study="finger-offsets"]').scroll_into_view_if_needed()
            local.wait_for_function("document.querySelector('[data-study=finger-offsets]').readyState>=2")
            local.close();checks.append('Direct file opening and local video playback')
            nojs=browser.new_context(java_script_enabled=False)
            fallback=nojs.new_page();fallback.goto(origin,wait_until='load')
            assert fallback.locator('.study-panel:visible').count()==4
            assert fallback.locator('.study-download:visible').count()==8
            nojs.close();checks.append('All eight video downloads remain available without JavaScript')
            browser.close()
        assert not errors,errors
        assert not missing,missing
        report={'checks':checks,'javascript_errors':errors,'failed_requests':missing}
        (OUT/'checks.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
    finally:server.shutdown()

if __name__=='__main__':main()
