"""Observe actual preparation owners; routed worker aborts can omit requestfailed."""
import asyncio
import time
from urllib.parse import urlsplit


async def workers(context,page):
    session=await context.new_cdp_session(page)
    try:
        origin=urlsplit(page.url)
        return [t['targetId'] for t in (await session.send('Target.getTargets'))['targetInfos']
            if t['type']=='worker' and urlsplit(t['url']).netloc==origin.netloc
            and urlsplit(t['url']).path.endswith('/course-image-worker.js')]
    finally:await session.detach()


async def acknowledged(context,page,owners,failed,name,started):
    deadline=started+1
    while time.perf_counter()<deadline:
        failures=[r for r in failed if f'/images/{name}-train.npz' in r[0] and r[1]>=started]
        current=await workers(context,page) if owners else []
        # A worker request intercepted by Playwright's routing layer may not
        # emit requestfailed when its owner closes. Require actual owner removal
        # instead of treating absence of that optional event as a product fault.
        if (owners and not set(owners).intersection(current)) or (not owners and failures):
            seconds=time.perf_counter()-started
            if seconds>1:break
            return dict(seconds=seconds,requests=failures,owning_worker_ids=owners,
                remaining_worker_ids=current,measurement='CDP target disappearance' if owners else 'requestfailed',
                assertion=dict(kind='workflow',expected=dict(cancelled_within_one_second=True),
                    observed=dict(cancelled_within_one_second=seconds<=1),matched=True))
        await asyncio.sleep(.01)
    raise AssertionError(f'Image preparation cancellation was not observed within one second: owners={owners}; remaining={await workers(context,page)}')
