"""Fetch what the base->Instruct transfer gate needs: the base host and the L8 dictionaries."""
import os, sys, time
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet OOMs the login node
from huggingface_hub import snapshot_download

JOBS = [
    ("OpenMOSS-Team/Llama3_1-8B-Base-LXTC-8x",
     ["Llama3_1-8B-Base-L8TC-8x/*"]),
    ("OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x",
     ["Llama3_1-8B-Base-L8R-8x/*"]),
    ("meta-llama/Llama-3.1-8B", None),             # the control host, ~16 GB
]
for repo, allow in JOBS:
    for attempt in range(1, 21):
        try:
            print(f"[e3] {repo} attempt {attempt}", flush=True)
            p = snapshot_download(repo_id=repo, allow_patterns=allow, max_workers=2)
            print(f"[e3] DONE {repo} -> {p}", flush=True)
            break
        except Exception as e:
            print(f"[e3] {repo} failed: {type(e).__name__}: {str(e)[:120]} — retry", flush=True)
            time.sleep(min(30 * attempt, 240))
    else:
        print(f"[e3] GAVE UP {repo}", flush=True); sys.exit(1)
print("[e3] all assets present", flush=True)
