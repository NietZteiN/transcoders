#!/bin/bash
#SBATCH --job-name=mi_env_rebuild
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_mi_env_rebuild.out
# T0.1 -- restore the pip layer of the transcoders-mi env from environment.lock.txt.
#
# WHY. The env directory exists with the right interpreter (Python 3.11.16) but NO pip packages:
# 696 MB of python + pip + packaging, no torch. The conda create succeeded 2026-08-03 on the retired
# A6000 host (data/env_create.log exits 0); the pip layer did not survive the move to juno. So the
# experiments this sub-project exists for (E1-E7) have never been runnable here.
#
# NON-DESTRUCTIVE ON PURPOSE. The interpreter is correct and matches environment.yml (python=3.11),
# so this installs INTO the existing prefix. Nothing is deleted -- CLAUDE.md requires a human in the
# loop for that, and there is nothing here that needs removing.
#
# The lock file is a full freeze (162 lines) including circuit-tracer pinned to a git sha, so this is
# reproducible rather than a fresh solve. PyPI and GitHub were both verified reachable before submit.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
cd "$PROJ"
E="$MI_ENV"
export LD_LIBRARY_PATH="$E/lib:${LD_LIBRARY_PATH:-}"
echo "# T0.1 env rebuild · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
echo "# prefix $E"; $E/bin/python --version
# Use `python -m pip`, never the $E/bin/pip console script: the conda-forge-built entry point carries
# its build-time feedstock path and dies with
#   OSError: [Errno 2] No such file or directory: '/home/conda/feedstock_root/build_artifacts/.../work'
# before installing anything (job 412659). The module form has no such wrapper.
sha256sum environment.lock.txt
echo "# packages before: $($E/bin/python -m pip list 2>/dev/null | tail -n +3 | wc -l)"

# circuit-tracer is a git dependency, and git must NOT see our LD_LIBRARY_PATH. Exporting the conda
# lib dir (which env.sh requires, for libstdc++/CXXABI_1.3.15) makes the system git-remote-https bind
# conda's OpenSSL against the system libldap and abort:
#   symbol lookup error: /lib64/libldap.so.2: undefined symbol: EVP_md2, version OPENSSL_3.0.0
# That killed the first attempt (job 412655) in 10 s. So: clone with a CLEAN environment, pin the sha,
# and install from the local checkout -- same commit, no network inside pip.
# The conda-forge base packages ship a dist-info/direct_url.json naming their BUILD directory
# (/home/conda/feedstock_root/...). pip stats every installed distribution's direct_url.json when it
# resolves a multi-package requirements file, so the missing path aborts the whole install before
# anything downloads -- jobs 412659 and 412663, both dead in ~14 s. A single-package install never
# triggers it, which is why the interactive probe looked fine. Renamed, never deleted (CLAUDE.md), and
# idempotent so a re-run is a no-op.
for D in "$E"/lib/python3.11/site-packages/*.dist-info; do
  if [ -f "$D/direct_url.json" ] && grep -q feedstock_root "$D/direct_url.json" 2>/dev/null; then
    mv "$D/direct_url.json" "$D/direct_url.json.disabled"
    echo "# neutralised stale direct_url.json in $(basename "$D")"
  fi
done

SRC=/work/jvl210002/src/circuit-tracer
CT_SHA=$(grep -oE 'circuit-tracer.git@[0-9a-f]{40}' environment.lock.txt | head -1 | cut -d@ -f2)
echo "# circuit-tracer pinned sha: $CT_SHA"
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$(dirname "$SRC")"
  env -u LD_LIBRARY_PATH git clone --filter=blob:none --quiet \
      https://github.com/safety-research/circuit-tracer.git "$SRC" || { echo "# FATAL: clone failed"; exit 4; }
fi
env -u LD_LIBRARY_PATH git -C "$SRC" fetch --quiet origin "$CT_SHA" 2>/dev/null || true
env -u LD_LIBRARY_PATH git -C "$SRC" checkout --quiet "$CT_SHA" || { echo "# FATAL: checkout $CT_SHA failed"; exit 4; }
echo "# checked out $(env -u LD_LIBRARY_PATH git -C "$SRC" rev-parse HEAD)"

# requirements with the git URL swapped for the local checkout; everything else byte-identical
REQ=$TMPDIR/mi_lock_local.txt
# Also strip LOCAL-PATH requirements. environment.lock.txt line 93 reads
#   packaging @ file:///home/conda/feedstock_root/build_artifacts/bld/rattler-build_packaging_.../work
# which is a `pip freeze` artefact: on the retired host `packaging` came from conda, and freeze records
# conda-installed distributions as a file:// URL pointing at their BUILD directory. That path exists on
# no machine, so the lock could never rebuild anywhere -- pip aborts the entire resolve before
# downloading anything (jobs 412659 / 412663 / 412671, each dead in ~15 s). These are conda-provided
# base packages and are already present in the prefix, so dropping the line is correct rather than
# merely expedient; the dry-run confirms the remaining 160 pins resolve.
grep -vE '^circuit-tracer @|@ file://' environment.lock.txt > "$REQ"
echo "# requirements: $(wc -l < environment.lock.txt) lines -> $(wc -l < "$REQ") after stripping git + local-path entries"
"$E/bin/python" -m pip install --no-input --disable-pip-version-check -r "$REQ"; rc=$?
echo "# pip (pypi layer) rc=$rc"
if [ $rc -eq 0 ]; then
  "$E/bin/python" -m pip install --no-input --disable-pip-version-check --no-deps "$SRC"; rc=$?
  echo "# pip (circuit-tracer from local checkout, --no-deps to preserve the pinned solve) rc=$rc"
fi
echo "# packages after: $($E/bin/python -m pip list 2>/dev/null | tail -n +3 | wc -l)"

echo "=== import check (the four that define this project) ==="
LD_LIBRARY_PATH="$E/lib" "$E/bin/python" - <<'PY'
import sys
ok = True
for m in ("torch", "transformers", "sae_lens", "transformer_lens", "nnsight", "circuit_tracer",
          "numpy", "pandas", "sklearn", "yaml"):
    try:
        mod = __import__(m)
        print(f"  {m:18s} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        ok = False
        print(f"  {m:18s} FAIL {type(e).__name__}: {str(e)[:70]}")
import torch
print(f"  torch.cuda available={torch.cuda.is_available()} (expected False on a CPU node) "
      f"· built for cuda {torch.version.cuda}")
sys.exit(0 if ok else 3)
PY
rc2=$?
echo "# import check rc=$rc2"
"$E/bin/python" -m pip check 2>&1 | tail -5
echo "# done rc=$((rc + rc2)) $(date -u +%FT%TZ)"
exit $((rc + rc2))
