#!/usr/bin/env bash
# Cluster environment for juno-l-01. Source this; do not execute it.
#
# WHY THIS FILE EXISTS. Every script in this project was written for
# csr-94608.utdallas.edu: paths under /data/jvl210002, conda envs at a fixed prefix, GPUs chosen by
# reading `nvidia-smi` because there was no scheduler. None of that is true here. juno is a SLURM
# cluster: the login node has no GPU and no `nvidia-smi`, conda arrives through a module, and cards
# are requested, not picked. Rather than sed the old paths in twelve places, everything that moved
# is named once, here.
#
# GPU POLICY INVERTS. On the old box the rule was "check nvidia-smi, take only idle cards, never
# launch onto a card another job is using" — because nothing protected you. Here SLURM owns
# allocation: ask for what you need with --gres and it is yours exclusively. Do NOT go looking for
# idle cards on juno; there are none to find from the login node.

export PROJ=/work/jvl210002/migration/transcoders
export REPL=/work/jvl210002/migration/allocation_replication
export CONDA_ROOT=/work/jvl210002/conda_envs
export NLA_ENV=$CONDA_ROOT/nla-mi
export CS_ENV=$CONDA_ROOT/codesteer

# HF_HOME already migrated; keep model caches and pip temp off the small $HOME NFS.
export HF_HOME=/work/jvl210002/migration/hf_home
export TMPDIR=/work/jvl210002/tmp_pip
export PIP_CACHE_DIR=/work/jvl210002/tmp_pip/pip-cache

# conda's defaults put both envs and package cache under $HOME/.conda — small NFS. Redirect both.
export CONDA_ENVS_DIRS=$CONDA_ROOT
export CONDA_PKGS_DIRS=/work/jvl210002/conda_envs/.pkgs

# JDK 21: P0.3 compiles and executes Java. System java is 17 and the module tree stops at 11.
export JAVA_HOME=/work/jvl210002/tools/jdk-21
[ -d "$JAVA_HOME" ] && export PATH="$JAVA_HOME/bin:$PATH"
export OBF_JAVA_CLASSPATH=$REPL/artifact/PromptSteering/java_compat

# SLURM defaults for this project's GPU work.
export SLURM_GPU_PARTITION=${SLURM_GPU_PARTITION:-h200}

load_conda() {
  source /etc/profile.d/lmod.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null
  module load miniconda/24.11.1
  # `module load` only puts conda on PATH; it does not define the shell function, so a batch
  # script gets "CondaError: Run 'conda init' before 'conda activate'" and — because the
  # activate fails without stopping the script — then runs the job under the SYSTEM python,
  # which has no torch. Failing that way costs a GPU allocation and looks like a code bug.
  eval "$(conda shell.bash hook)"
}

# Activate or die. `conda activate` returning non-zero must not be allowed to fall through to a
# python that happens to exist; every caller wants the env or nothing.
activate_env() {
  local env_path="$1"
  conda activate "$env_path" || { echo "FATAL: could not activate $env_path" >&2; exit 90; }
  local want="$env_path/bin/python"
  [ "$(command -v python)" = "$want" ] || {
    echo "FATAL: python is $(command -v python), expected $want" >&2; exit 91; }
}
