# 02 — Environment: what will break on a new cluster

Read before running anything. Most of these are silent failures, not crashes.

---

## Two conda environments, and they are NOT interchangeable

| env | torch | transformers | used by |
|---|---|---|---|
| `nla-mi` | 2.11.0+cu130 | **5.12.1** | everything NLA: `layer_rotation.py`, `steer_run.py`, `steer_stats.py`, the AV/AR checkpoints |
| `codesteer` | 2.9.1+cu128 | **4.57.1** | attention steering: `allocation_replication/artifact/obfuscation/main.py`, all of P0.3 |

**They cannot be merged.** The attention backend imports transformers-4.x decoder internals
(`Qwen2Attention`, `apply_rotary_pos_emb`, `repeat_kv`) that do not exist in 5.x. This is why B5
had to move the NLA write hook *into* the codesteer env rather than the other way around — and why
`nla/src/steer.py` deliberately imports **only torch and stdlib**, resolving decoder layers by
attribute path (`model.layers` / `model.model.layers` / `transformer.h`) so nothing binds it to a
transformers version. **Keep it that way.**

Spec files: `transcoders/environment.yml` + `environment.lock.txt`, and `transcoders/nla/environment.yml`.

> **Known gap in the spec:** the SGLang server path needs `ninja` (for flashinfer JIT). It was
> installed ad hoc and never added to `nla/environment.yml`. Add it if you rebuild.

---

## Paths that are hard-coded and will not exist elsewhere

Everything is anchored under `/data/jvl210002/`. Grep for it before the first run:

```bash
grep -rn "/data/jvl210002" nla/scripts/ nla/src/ | grep -v "^Binary"
```

Known hard-coded sites:

| what | where | note |
|---|---|---|
| project roots | `nla/scripts/p0_autopilot.sh`, `p03_site.sh` | `PROJ` and `REPL` at the top |
| conda envs | same two scripts | `NLA_ENV`, `CS_ENV` |
| `HF_HOME` | scripts + shell | `/data/jvl210002/my_downloads/.cache/huggingface` |
| `TMPDIR` | scripts + shell | `/data/jvl210002/tmp_pip` — keep temp off small NFS `$HOME` |
| **JDK 21** | `p03_site.sh` | `JAVA_HOME=/data/jvl210002/my_downloads/tools/jdk-21.0.4+7` — **P0.3 executes Java**, it will fail without a JDK |
| `OBF_JAVA_CLASSPATH` | `p03_site.sh` | `$REPL/artifact/PromptSteering/java_compat` |
| **GPU ids** | `p03_site.sh` | hard-codes **GPUs 0 and 3**; edit for the new cluster |

---

## Hardware assumptions

The origin box: 4 × RTX A6000 (48 GB), 96 cores, ~250 GB RAM, **no scheduler**. Every script
selects GPUs itself and the project rule is: check `nvidia-smi`, use only idle cards, pin with
`CUDA_VISIBLE_DEVICES` **before importing torch**, and never launch onto a card another job is
using. If the new cluster *has* a scheduler (SLURM etc.), the scripts do not know about it and
will need wrapping.

Memory: the 7B subject in bf16 is ~15 GB; loading the AR alongside pushes a P0.2-style run to
~26 GB. Both fit one 48 GB card. Note the banked observation that **the AR pins ~11 GB after its
caches are built — `del ar` returns ~12 GB of headroom.**

---

## Model weights

**Do not copy the 25 GB checkpoint directory.** Re-download:

```
kitft/nla-qwen2.5-7b-L20-av   ->  transcoders/nla/data/checkpoints/av/
kitft/nla-qwen2.5-7b-L20-ar   ->  transcoders/nla/data/checkpoints/ar/
```

Both are frozen released checkpoints (Apache-2.0, fine-tuned from `Qwen2.5-7B-Instruct`).
Method paper: *Natural Language Autoencoders Produce Unsupervised Explanations of LLM
Activations*, transformer-circuits.pub, 2026.

Subject models pulled from HF: `Qwen/Qwen2.5-7B-Instruct` (NLA side) and
`Qwen/Qwen2.5-Coder-7B-Instruct` (attention-steering side, P0.3/B5).

> **Why two subject models, and don't "fix" it.** B1 established that the frozen NLA reads the
> Coder sibling at rt_cos **0.694** vs 0.864 on its host — a **failed** transfer gate, and the loss
> is *directional* not scalar (injection normalizes magnitude away, so re-deriving the injection
> scale is a provable no-op). Attention steering, meanwhile, was only ever replicated on the Coder.
> Composition therefore forces one model and no choice leaves both arms clean. This is a known,
> documented limitation, not an oversight.

---

## A patch that lives outside this repo

`transformers 5.12` broke the reference `apply_chat_template` return contract. The fix is
`nla/patches/transformers512_chat_template.patch`, applied at 2 call sites — it is part of why G0
replicates exactly (0/101 over tolerance). If the new cluster builds a different transformers
version, re-check that gate before trusting any read.

---

## First-run verification, in order

Do these before any real job. Each is minutes and each has caught a real bug here.

```bash
# 1. layer indexing — must print exactly 0.0e+00 relative error
CUDA_VISIBLE_DEVICES=<free> $NLA_ENV/bin/python nla/src/layer_rotation.py --limit 2 --gate-only

# 2. the cross-repo flag exists and targets an arbitrary band
#    expect: [Steering] explicit layer band: start=20, end=20 (num_layers=28)
grep -n "steer-layers" ../allocation_replication/artifact/obfuscation/main.py

# 3. the test suite
$NLA_ENV/bin/python -m pytest nla/tests/ -q      # 86 Python tests across 5 modules
```

The JS regression tests (`nla/tests/artifact_*.mjs`) extract the **shipped** page code and run it
rather than re-implementing it, so they cannot drift from what actually ships. Run them from the
directory holding `nla_results.html`.
