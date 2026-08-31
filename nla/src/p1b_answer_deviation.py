"""Answer-length DEVIATION control.

Answer character length at `last_prompt` came back at rho = 0.6966 — but the expected answer's
length is a property of the item, readable from the code itself, so that probe may be decoding the
TASK rather than a decided answer. The deviation |len(answer) - len(truth)| is not knowable from
the prompt: it requires knowing what the model will actually emit. If THAT is decodable only at
the answer line, the late-readout reading holds; if it is decodable at the prompt too, the model
commits to its answer before generating.
"""
import json, random, statistics as st, sys
from pathlib import Path
import numpy as np
P = Path("/work/jvl210002/migration/transcoders")
sys.path.insert(0, str(P / "nla/src")); sys.path.insert(0, str(P / "nla/vendor/nla-repo"))
from p1b_graded_labels import oof_ridge, spearman
from steer_run import SEED, load_pairs

truth = {p["snippet_id"]: str(p["truth"]) for p in load_pairs(None, random.Random(SEED))}
z = np.load(P / "data/nla/p0/p1b/pos_acts.npz", allow_pickle=True)
acts, valid, groups = z["acts"], z["valid"], z["groups"]
positions = [str(x) for x in z["positions"]]
replies = {r["snippet_id"]: r for r in json.load(open(P / "data/nla/p0/p1b/position_depth.json"))["replies"]}

dev, tl = [], []
for sid in groups:
    a = (replies.get(str(sid)) or {}).get("answer")
    t = truth.get(str(sid))
    dev.append(abs(len(a) - len(t)) if (a and t) else np.nan)
    tl.append(len(t) if t else np.nan)
dev, tl = np.array(dev, float), np.array(tl, float)
print(f"n with both answer and truth: {int((~np.isnan(dev)).sum())} · mean |dev| {np.nanmean(dev):.2f}")

out = {}
for tag in ("last_prompt", "answer_line"):
    pi = positions.index(tag)
    m = valid[:, pi] & ~np.isnan(dev)
    X, y, g = acts[m][:, pi], dev[m], np.array([str(x) for x in groups[m]])
    curve = [round(spearman(y, oof_ridge(X[:, L], y, g)), 4) for L in range(acts.shape[2])]
    best = int(np.argmax(curve))
    # the item's own truth length is the confound this control removes; report what it buys alone
    base = round(spearman(y, oof_ridge(tl[m].reshape(-1, 1), y, g)), 4)
    out[tag] = {"n": int(m.sum()), "argmax_layer": best, "argmax_rho": curve[best],
                "mean_rho": round(st.mean(curve), 4), "truth_length_baseline_rho": base,
                "rho_by_layer": curve}
    print(f"[dev] |answer-truth| @ {tag:<12} argmax L{best} {curve[best]:+.4f} · "
          f"mean {st.mean(curve):+.4f} · truth-length baseline {base:+.4f}")
d = out["answer_line"]["argmax_rho"] - out["last_prompt"]["argmax_rho"]
out["delta"] = round(d, 4)
out["reading"] = ("SUPPORTS LATE READOUT — what the model will actually emit is decodable only "
                  "once it is committed" if d >= 0.15 else
                  "ANSWER COMMITTED EARLY — the emitted answer is already decodable at the prompt")
print("\n[dev] delta %+.4f -> %s" % (d, out["reading"]))
json.dump(out, open(P / "data/nla/p0/p1b/answer_deviation_control.json", "w"), indent=2)
