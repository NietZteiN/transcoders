#!/bin/bash
#SBATCH --job-name=nla_r1_rpacks
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/work/jvl210002/migration/transcoders/log/slurm/%j_nla_r1_renamed_packs.out
# H-R1b -- rebuild case packs on the ADVERSARIALLY RENAMED Java corpus, because their obfuscation
# runner rebuilds the pack from the obfuscated variant (obfuscation/main.py:428). This is what puts
# the decoy method name inside every case expression the model is asked to evaluate.
# Amendment: log/nla-harness/2026-09-14_ase-prereg-amendment.md. CPU only (JDK, no GPU).
set -uo pipefail
source /work/jvl210002/migration/transcoders/nla/scripts/juno_env.sh
load_conda; activate_env "$NLA_ENV"; cd "$PROJ"
export TOKENIZERS_PARALLELISM=false PYTHONHASHSEED=0 PYTHONUNBUFFERED=1
echo "# H-R1b renamed case packs · job $SLURM_JOB_ID on $SLURMD_NODENAME · $(date -u +%FT%TZ)"
sha256sum nla/src/ase_casepacks.py nla/src/ase_rename_java.py
java -version 2>&1 | head -1
python nla/src/ase_casepacks.py \
  --lang-dir /scratch/juno/jvl210002/ase2026/renamed_humaneval_java \
  --cache-dir /scratch/juno/jvl210002/ase2026/cache_cases_renamed \
  --out /scratch/juno/jvl210002/ase2026/packs_renamed_humaneval_java.jsonl \
  --max-hours 3.5; rc=$?
echo "# done rc=$rc $(date -u +%FT%TZ)"
exit $rc
