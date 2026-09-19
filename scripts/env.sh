# Source this to enter the project environment:   source scripts/env.sh
# (Not executable on purpose — it must be sourced to affect your shell.)
#
# Sets up the transcoders-mi conda env + the env vars ../CLAUDE.md §2 requires.
#
# PATHS REBASED 2026-09-19 for juno. Every path here named the retired csr-94608 host
# (/data/jvl210002/...), so sourcing this pointed TRANSCODERS_ENV at a prefix that does not exist
# and silently left you on the system python. The env itself is at $CONDA_ROOT/transcoders-mi and
# is also exported as MI_ENV by nla/scripts/juno_env.sh, which is the cluster-wide script.
# NOTE: LD_LIBRARY_PATH must include $ENV/lib BEFORE python starts — the conda libstdc++
# is newer than the system one (CXXABI_1.3.15); without it, sqlite3/ICU imports fail.
# (Same fix the sibling translation project uses.)

export TRANSCODERS_ENV=/work/jvl210002/conda_envs/transcoders-mi
export LD_LIBRARY_PATH="$TRANSCODERS_ENV/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR=/work/jvl210002/tmp_pip
export HF_HOME="${HF_HOME:-/work/jvl210002/migration/hf_home}"
export PATH="$TRANSCODERS_ENV/bin:$PATH"

echo "transcoders-mi env active: $(python --version 2>&1) @ $TRANSCODERS_ENV"
