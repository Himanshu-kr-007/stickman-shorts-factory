# usage: python3 gen_images.py job_dir
# Makes every still listed in job_dir/*.json with Cloudflare Workers AI (FLUX.1 schnell) on the free daily allowance
# (10,000 neurons/day ~ 100+ images). Needs env CF_ACCOUNT_ID and CF_API_TOKEN (GitHub repo secrets).
# Each still segment: {"type": "still", "path": "a1.png", "prompt": "...", "seed": 11}. Existing files are kept.
import sys, os, json, glob, time, base64, io, urllib.request, urllib.error
from PIL import Image

STYLE = ("Polished 2D cartoon illustration from a viral stickman comedy animation: clean bold black outlines, flat cel shading, "
         "rich painted full-color background, warm cinematic light, {FRAME}. Stickman characters with large solid white round heads "
         "with a thin black outline, big expressive white eyes with small black pupils, black eyebrow strokes, a small expressive mouth, "
         "thin black stick arms and legs, small black rounded hands and feet. No text, no letters, no numbers, no captions, no watermark. SCENE: ")
FRAME = {'9:16': 'subject centered in the middle third of the picture with empty space above and below, tall composition',
         '16:9': 'wide cinematic composition with the subject in the middle'}
ACC, TOK = os.environ.get('CF_ACCOUNT_ID', '').strip(), os.environ.get('CF_API_TOKEN', '').strip()
MODEL = os.environ.get('CF_IMAGE_MODEL', '@cf/black-forest-labs/flux-1-schnell')
if not ACC or not TOK:
    print('MISSING_SECRETS: add CF_ACCOUNT_ID and CF_API_TOKEN as GitHub repository secrets'); sys.exit(2)
URL = f'https://api.cloudflare.com/client/v4/accounts/{ACC}/ai/run/{MODEL}'

def fetch(prompt, seed, path, aspect):
    body = json.dumps({'prompt': (STYLE.replace('{FRAME}', FRAME.get(aspect, FRAME['9:16'])) + prompt)[:2048], 'steps': 8, 'seed': int(seed)}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, data=body, headers={'Authorization': f'Bearer {TOK}', 'Content-Type': 'application/json'})
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            img = (r.get('result') or {}).get('image')
            if not img: raise ValueError(str(r.get('errors'))[:200])
            im = Image.open(io.BytesIO(base64.b64decode(img))).convert('RGB'); im.save(path)
            print('ok', os.path.basename(path), im.size, flush=True); return True
        except urllib.error.HTTPError as e:
            msg = e.read()[:300].decode('utf8', 'replace'); print('retry', os.path.basename(path), attempt, e.code, msg, flush=True)
            if e.code in (401, 403): sys.exit(3)  # bad token: retrying will not help
            if e.code == 429 and 'neuron' in msg.lower(): print('DAILY_FREE_LIMIT_REACHED'); sys.exit(4)
            time.sleep(10 + attempt * 10)
        except Exception as e:
            print('retry', os.path.basename(path), attempt, repr(e)[:200], flush=True); time.sleep(10 + attempt * 10)
    return False

failed = []
for spec in sorted(glob.glob(os.path.join(sys.argv[1], '*.json'))):
    s = json.load(open(spec))
    for i, sg in enumerate(s.get('segments', [])):
        if sg.get('type') != 'still' or not sg.get('prompt'): continue
        path = os.path.join(sys.argv[1], sg['path'])
        if os.path.exists(path): continue
        if not fetch(sg['prompt'], sg.get('seed', 1000 + i), path, s.get('aspect', '9:16')): failed.append(sg['path'])
if failed:
    print('FAILED', failed); sys.exit(1)
