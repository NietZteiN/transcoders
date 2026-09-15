#!/bin/bash
#SBATCH --job-name=nla_r1_packs
#SBATCH --partition=normal,h200,h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r1_casepacks.out
# H-R1a stage 1 -- build execution-validated T/F case packs for the 164 HumanEval-X Java snippets
# using the ASE-2026 artifact's own counterfactual_eval.build_case_pack. CPU ONLY (needs a JDK, no GPU).
# Pre-registered in log/nla-harness/2026-09-14_ase-replication-prereg.md.
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# H-R1a case packs · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_casepacks.py
java -version 2>&1 | head -1
python -c "import javalang; print('javalang', javalang.__name__, 'OK')"
python nla/src/ase_casepacks.py --max-hours 3.5; rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
