"""Resilient fetch of the Gemma-3-12B NLA pair.

Unauthenticated Hub requests are rate-limited — the first attempt died on a 504 from the
xet-read-token endpoint followed by a ReadTimeout, ~9.8 GB in. `snapshot_download` resumes from
whatever is already on disk, so the fix is simply to keep asking. A token would raise the limits
and is worth having, but the pair is ungated so it is not required.
"""
import sys, time
from huggingface_hub import snapshot_download

REPOS = ["kitft/nla-gemma3-12b-L32-av", "kitft/nla-gemma3-12b-L32-ar"]
MAX_TRIES = 40

for repo in REPOS:
    for attempt in range(1, MAX_TRIES + 1):
        try:
            print(f"[fetch] {repo} attempt {attempt}", flush=True)
            path = snapshot_download(repo_id=repo, max_workers=4)
            print(f"[fetch] DONE {repo} -> {path}", flush=True)
            break
        except Exception as e:
            wait = min(30 * attempt, 300)
            print(f"[fetch] {repo} attempt {attempt} failed: {type(e).__name__}: "
                  f"{str(e)[:160]} — retrying in {wait}s", flush=True)
            time.sleep(wait)
    else:
        print(f"[fetch] GAVE UP on {repo} after {MAX_TRIES} attempts", flush=True)
        sys.exit(1)
print("[fetch] all repos complete", flush=True)
