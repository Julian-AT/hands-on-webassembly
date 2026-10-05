"""Inject real startup request failures, then exercise the visible Retry button."""
import argparse
import time
from playwright.sync_api import sync_playwright, expect
from common import PROOF, write_json
from provenance import browser_stamp
from browser_startup import application_ready


def stamp():
    return browser_stamp('recovery')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser', choices=['chrome', 'firefox', 'webkit'], default='chrome')
    args = parser.parse_args()
    result = {'browser': args.browser, 'distribution': 'installed' if args.browser == 'chrome' else 'playwright',
              'provenance': stamp(), 'cases': {}, 'status': 'fail'}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome') if args.browser == 'chrome' else getattr(p, args.browser).launch()
        result['version'] = browser.version
        for asset in ('app.json', 'pyodide-worker.js', 'pyodide.asm.wasm'):
            context = browser.new_context()
            page = context.new_page()
            failed = []
            responses = []
            page.on('response', lambda response: responses.append({'url': response.url, 'via_service_worker': response.from_service_worker}) if '.wasm' in response.url else None)
            def abort(route):
                failed.append(route.request.url)
                route.abort('connectionfailed')
            context.route('**/' + asset, abort)
            start = time.monotonic()
            try:
                page.goto('http://127.0.0.1:8008/unit1/', wait_until='domcontentloaded')
                expect(page.locator('#course-startup')).to_be_visible()
                retry = page.get_by_role('button', name='Reload to retry')
                expect(retry).to_be_visible(timeout=30000)
                elapsed = time.monotonic() - start
                assert failed, 'The selected download was not actually interrupted'
                message = page.locator('#course-startup').inner_text()
                context.unroute('**/' + asset, abort)
                retry.click()
                application = application_ready(page)
                expect(page.locator('#course-startup')).to_be_hidden(timeout=10000)
                assert responses and all(not response['via_service_worker'] for response in responses), 'Ordinary WASM still depends on a service worker'
                result['cases'][f'unit1/{asset}/interrupted-retry'] = {'status': 'pass', 'error_seconds': elapsed,
                    'failed_requests': failed, 'message': message, 'wasm_responses': responses,
                    'application': application}
            except Exception as error:
                result['cases'][f'unit1/{asset}/interrupted-retry'] = {'status': 'fail', 'error': str(error),
                    'failed_requests': failed, 'text': page.locator('body').inner_text()}
            finally:
                context.close()
            print(args.browser, asset, result['cases'][f'unit1/{asset}/interrupted-retry']['status'], flush=True)
        browser.close()
    result['status'] = 'pass' if all(case['status'] == 'pass' for case in result['cases'].values()) else 'fail'
    if result['provenance']['fingerprint'] != stamp()['fingerprint']:
        result['original_status'] = result['status']
        result['status'] = 'stale'
    write_json(PROOF / f'evidence/startup-recovery-{args.browser}.json', result)
    return int(result['status'] != 'pass')


if __name__ == '__main__':
    raise SystemExit(main())
