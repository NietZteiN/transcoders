# artifacts/ — small files that must survive even if nothing else does

Copies, not originals. Each has a canonical home in the repo; these exist so the handoff folder is
self-contained if the working tree or the cross-repo commit fails to travel.

| file | canonical home | why it is here |
|---|---|---|
| `2026-08-27_p0-triage-prereg.md` | `log/nla-harness/` | the **frozen decision rules**. Phase-0's results are meaningless without them, and they must not be renegotiated after the numbers land. |
| `layer_rotation.json` | `data/nla/p0/` | the **complete P0.1 result** — verdict HARD, all 28 layers, rotation + leave-one-out coherence + relative magnitude. Not reproducible without a GPU. |
| `p03_partial_cells.json` | `data/nla/p0/p03/` | the **8 finished P0.3 cells** at full scale (492 runs / 5,790 cases each), plus the two partial ones. |
| `steer_layers_flag.patch` | `allocation_replication/artifact/obfuscation/main.py` | **lives in a different repo** and is the easiest thing here to lose. P0.3 cannot target layer 20 without it. |

## Applying the patch

```bash
cd <path>/allocation_replication
git apply <path>/transcoders/nla/continuation/artifacts/steer_layers_flag.patch
# verify:
grep -n "steer-layers" artifact/obfuscation/main.py
```

`--steer-last-n-layers` is a *suffix* wrapper: `n=1` gives layer **27**, not 20. The patch adds
`--steer-layers A:B` for an explicit inclusive band, which is what P0.3 needs. The underlying
config fields (`steer_layer_start` / `steer_layer_end`) always existed; only the CLI was
suffix-shaped.
