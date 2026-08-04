# Source this to enter the project environment:   source scripts/env.sh
# (Not executable on purpose — it must be sourced to affect your shell.)
#
# Sets up the transcoders-mi conda env + the env vars ../CLAUDE.md §2 requires.
# NOTE: LD_LIBRARY_PATH must include $ENV/lib BEFORE python starts — the conda libstdc++
# is newer than the system one (CXXABI_1.3.15); without it, sqlite3/ICU imports fail.
# (Same fix the sibling translation project uses.)

export TRANSCODERS_ENV=/data/jvl210002/conda_envs/transcoders-mi
export LD_LIBRARY_PATH="$TRANSCODERS_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR=/data/jvl210002/tmp_pip
export HF_HOME="${HF_HOME:-/data/jvl210002/my_downloads/.cache/huggingface}"
export PATH="$TRANSCODERS_ENV/bin:$PATH"

echo "transcoders-mi env active: $(python --version 2>&1) @ $TRANSCODERS_ENV"
