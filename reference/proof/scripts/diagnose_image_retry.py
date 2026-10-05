"""Retain actual request/output traces around one corrupt-pair retry."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from playwright.async_api import async_playwright,expect
from browser_html_proxy import site_manifest
from common import ROOT,PROOF,sha,write_json


async def run(args,result,output):
    async with async_playwright() as p:
        browser=await p.chromium.launch(channel='chrome' if args.browser=='chrome' else 'msedge')
        result['version']=browser.version
        context=await browser.new_context();page=await context.new_page()
        trace=[];console=[];result.update(trace=trace,console=console)
        page.on('console',lambda message:console.append(dict(type=message.type,text=message.text)))
        context.on('request',lambda request:trace.append(dict(event='request',url=request.url,time=time.perf_counter())) if '/images/' in request.url else None)
        context.on('requestfailed',lambda request:trace.append(dict(event='failed',url=request.url,error=request.failure,time=time.perf_counter())))
        context.on('response',lambda response:trace.append(dict(event='response',url=response.url,status=response.status,headers=response.headers,time=time.perf_counter())) if '/images/' in response.url else None)
        try:
            await page.goto(f'http://127.0.0.1:{args.port}/unit6/',wait_until='domcontentloaded')
            app=page.frame_locator('iframe')
            await expect(app.locator('#dataset.shiny-bound-input')).to_be_attached(timeout=120000)
            await app.get_by_role('tab',name='FNN: Data',exact=True).click()
            async def corrupt(route):await route.fulfill(status=200,body=b'corrupt test split')
            await context.route('**/images/MNIST-test.npz',corrupt)
            await app.locator('#load').click()
            await expect(app.locator('#data_info')).to_contain_text('checksum',timeout=120000)
            result['before_retry']=await app.locator('#data_info').inner_text()
            await context.unroute('**/images/MNIST-test.npz',corrupt)
            await app.locator('#load').click()
            try:
                await expect(app.locator('#data_info')).to_contain_text('Loaded MNIST data.',timeout=30000)
                result['retry_observation']='completed'
            except Exception as error:result.update(retry_observation='failed',error=str(error))
            result['after_retry']=await app.locator('#data_info').inner_text()
            result['input_values']=await app.locator('body').evaluate('()=>window.Shiny.shinyapp.$inputValues')
            result['status']='pass'
        except Exception as error:result.update(status='fail',error=str(error))
        finally:await browser.close();write_json(output,result)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--site',type=Path,required=True)
    parser.add_argument('--browser',choices=('chrome','edge'),required=True);parser.add_argument('--port',type=int,default=8077)
    parser.add_argument('--label',required=True);args=parser.parse_args()
    output=PROOF/f'evidence/image-retry-diagnostic-{args.browser}-{args.label}.json'
    if output.exists():raise FileExistsError('Retain previous diagnostic; choose a fresh label')
    result=dict(status='running',browser=args.browser,distribution='installed',inputs=site_manifest(args.site),
        executor_sha256=sha(Path(__file__)),scope='Network/output diagnostic for corrupt-pair retry; no acceptance or reliability claim.')
    server=subprocess.Popen(['node',str(ROOT/'tools/preview.mjs'),str(args.site)],env=dict(os.environ,PORT=str(args.port)),stdout=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if server.poll() is not None:raise RuntimeError('Diagnostic server exited')
            try:urllib.request.urlopen(f'http://127.0.0.1:{args.port}/unit6/',timeout=1).close();break
            except OSError:time.sleep(.1)
        asyncio.run(run(args,result,output))
    finally:server.terminate();server.wait(timeout=10)
    print(result['status'],result.get('retry_observation'),flush=True)


if __name__=='__main__':main()
