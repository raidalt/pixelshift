"""Bounded native-work concurrency; never queue a whole photo sequence in RAM."""
import os,subprocess,sys
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from collections import deque

@lru_cache(maxsize=1)
def physical_memory():
    try:return int(os.sysconf('SC_PHYS_PAGES'))*int(os.sysconf('SC_PAGE_SIZE'))
    except (AttributeError,ValueError,OSError):pass
    if sys.platform=='darwin':
        result=subprocess.run(['/usr/sbin/sysctl','-n','hw.memsize'],capture_output=True,text=True)
        try:return int(result.stdout.strip())
        except ValueError:pass
    return 0

def workers(estimated_bytes=0):
    requested=os.environ.get('PIXELSHIFT_WORKERS')
    if requested is not None:
        try:return max(1,min(2,int(requested)))
        except ValueError:return 1
    memory=physical_memory()
    return 2 if (os.cpu_count() or 1)>=4 and memory>=16*1024**3 and estimated_bytes*2<=memory//4 else 1

def ordered_map(function,items,count=1,check_cancel=lambda:None):
    if count<=1:
        for item in items:check_cancel();yield function(item)
        return
    iterator=iter(items);pool=ThreadPoolExecutor(max_workers=count,thread_name_prefix='pixelshift');pending=deque()
    try:
        for _ in range(count):
            try:item=next(iterator)
            except StopIteration:break
            pending.append(pool.submit(function,item))
        while pending:
            check_cancel();result=pending.popleft().result();check_cancel()
            try:item=next(iterator)
            except StopIteration:pass
            else:pending.append(pool.submit(function,item))
            yield result
    finally:
        for future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
