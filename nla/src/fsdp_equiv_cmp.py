"""Compare the two arms of the FSDP equivalence gate. CPU only; reads two train_log.jsonl files.

THE GATE. FSDP was chosen over 8-bit Adam / bf16 weights precisely because it leaves the optimizer
and precision recipe intact, so 12B pairs stay comparable with the 33 existing 4B pairs and with
the released kitft/nla-gemma3-12b-L32 pair. That argument is worthless unless the multi-GPU path is
verified to be the same method -- which is what this checks.

THE SHARPEST DIAGNOSTIC IS grad_norm AT STEP 1. Both likely bugs show up there, scaled:
  * the normalizer trap (each rank dividing by the GLOBAL token count while FSDP *averages*
    gradients) scales every gradient by 1/world          -> ratio ~ 1/world
  * per-shard gradient clipping instead of the global norm -> ratio ~ 1/sqrt(world)
Step-1 LOSS is computed before any update, so it validates the data partition and accumulation
order only; later steps test that the trajectories stay together.

Tolerances (declared before any output was seen, see nla/scripts/nla_fsdp_equiv.sh header):
  step-1 loss      rel < 1e-4   identical weights; only fp summation order differs
  step-1 grad_norm rel < 5e-3   catches the 1/world and 1/sqrt(world) signatures
  steps 2..N       mean rel < 2e-2, final < 3e-2
                   (fused AdamW is unavailable for DTensor -> foreach under --fsdp: same AdamW,
                    not bit-identical, so trajectories drift slightly)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def rows(root: Path, K: int) -> dict[int, dict]:
    p = root / f"L{K}/av/train_log.jsonl"
    if not p.exists():
        sys.exit(f"missing {p} -- did that arm finish?")
    return {d["step"]: d for d in map(json.loads, open(p))}


def main() -> int:
    a, b, K, W = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    A, B = rows(a, K), rows(b, K)
    steps = sorted(set(A) & set(B))
    if not steps:
        sys.exit("no common steps logged")
    rel = lambda x, y: abs(x - y) / max(abs(x), abs(y), 1e-12)  # noqa: E731
    s1 = steps[0]
    l1 = rel(A[s1]["loss"], B[s1]["loss"])
    g1 = rel(A[s1]["grad_norm"], B[s1]["grad_norm"])
    ratio = B[s1]["grad_norm"] / A[s1]["grad_norm"] if A[s1]["grad_norm"] else float("nan")
    losses = [rel(A[s]["loss"], B[s]["loss"]) for s in steps]
    mean_rel, last_rel = sum(losses) / len(losses), losses[-1]

    print(f"world={W}  common steps {steps[0]}..{steps[-1]} (n={len(steps)})")
    print(f"  step {s1}: loss      A={A[s1]['loss']:.6f}  B={B[s1]['loss']:.6f}  rel={l1:.2e}")
    print(f"  step {s1}: grad_norm A={A[s1]['grad_norm']:.6f}  B={B[s1]['grad_norm']:.6f}  rel={g1:.2e}  ratio={ratio:.4f}")
    for s in steps:
        print(f"    step {s:>4}  A {A[s]['loss']:.6f}  B {B[s]['loss']:.6f}  rel {rel(A[s]['loss'], B[s]['loss']):.2e}")
    for r, nm in ((a, "A"), (b, "B")):
        mem = max(d.get("mem_gb", 0) for d in rows(r, K).values())
        print(f"  peak mem arm {nm}: {mem:.1f} GB/GPU")

    ok = True
    def chk(name, val, lim):
        nonlocal ok
        good = val < lim
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: {val:.2e} < {lim:.0e}")
    chk("step-1 loss rel", l1, 1e-4)
    chk("step-1 grad_norm rel", g1, 5e-3)
    chk("mean rel loss diff", mean_rel, 2e-2)
    chk("final rel loss diff", last_rel, 3e-2)
    if abs(ratio - 1.0 / W) < 0.06:
        print(f"  DIAGNOSIS: grad_norm ratio ~1/{W} -> the NORMALIZER TRAP (missing scale_for_world).")
    elif abs(ratio - W ** -0.5) < 0.06:
        print(f"  DIAGNOSIS: grad_norm ratio ~1/sqrt({W}) -> clipping is seeing per-shard norms.")
    print("EQUIVALENCE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
