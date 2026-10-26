# usage: python3 gen_images.py job_dir
# Makes every still listed in job_dir/*.json with a free, key-less image service (Pollinations, Flux model).
# Each still: {"image": "a1.png", "prompt": "..."}; an existing file is kept (re-runs are cheap).
import sys, os, json, glob, time, urllib.parse, urllib.request, io
from PIL import Image

STYLE = ("Vertical 9:16 still frame from a polished 2D cartoon stickman comedy short: clean bold black outlines, soft cel shading, "
         "rich painted full-color background, warm cinematic light. Stickmen have large solid white round heads with a thin black outline, "
         "big expressive white eyes with small black pupils, black eyebrow strokes, a small mouth, thin black stick arms and legs, "
         "small black rounded hands and feet. No text, no letters, no numbers, no captions, no watermark. SCENE: ")
job = sys.argv[1]

def fetch(prompt, seed, path):
    q = urllib.parse.quote(STYLE + prompt, safe='')
    url = f"https://image.pollinations.ai/prompt/{q}?width=768&height=1376&seed={seed}&model=flux&nologo=true&private=true"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'stickman-shorts-factory/1.0'})
            data = urllib.request.urlopen(req, timeout=180).read()
            im = Image.open(io.BytesIO(data)).convert('RGB')
            if im.width < 300 or im.height < 500: raise ValueError(f'too small {im.size}')
            im.save(path); print('ok', os.path.basename(path), im.size, flush=True); return True
        except Exception as e:
            print('retry', os.path.basename(path), attempt, repr(e)[:160], flush=True); time.sleep(10 + attempt * 15)
    return False

failed = []
for spec in sorted(glob.glob(os.path.join(job, '*.json'))):
    s = json.load(open(spec))
    for i, sg in enumerate(s.get('segments', [])):
        if sg.get('type') != 'still' or not sg.get('prompt'): continue
        path = os.path.join(job, sg['path'])
        if os.path.exists(path): continue
        if not fetch(sg['prompt'], sg.get('seed', 1000 + i), path): failed.append(sg['path'])
        time.sleep(4)  # be gentle with the free service
if failed:
    print('FAILED', failed); sys.exit(1)
