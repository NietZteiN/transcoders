"""Judge-free grading of what an NLA read claims a program computes.

WHY THIS IS NOT AN LLM JUDGE. This project already ran that experiment. N7 spent two GPUs and
5,653 judged comparisons on a read-vs-text alignment score; it separated real from shuffled
pairs at AUC 0.757 and still could not rank a single item (kappa 0.049 against a second rater,
rho 0.032 against the free AR baseline), and the hypothesis it was built to test went
unadjudicated. What worked instead, in N9, was a regex against a ground-truth field the
verbalizer never sees — it labelled all 9,511 readings for free and produced the sharpest
result in the programme.

The adversarial-rename design hands us exactly such a field. Condition C2 injects a **known**
wrong algorithm name, chosen by us. So "does the read assert the injected algorithm" is a
closed-set membership question with a ground truth by construction, answerable by regex over a
synonym set. No judge, no rubric, no calibration set, no inter-rater kappa.

WHAT THIS CANNOT DO, STATED PLAINLY. A closed vocabulary cannot notice an algorithm it does not
list, so `ALGO_NONE` means "nothing in the vocabulary fired", never "the read is contentless".
Recall against the vocabulary should be measured once on a sample and reported.

THE STANDING RISK THIS MODULE IS FIGHTING. NLA reads are theme-reliable and
specifics-confabulated: 37% of this project's banked readings named the wrong *language*, and
libraries and frameworks are invented wholesale. "Which algorithm is this" is a **specific**
claim, which is the class that confabulates. Hence `confabulation_floor` — the C1 (neutral
rename) rate with no false content available to echo — is not a nicety, it is the denominator
that decides whether any C2 number means anything.

Everything here is pure CPU and deterministic.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

# ── The closed algorithm vocabulary ─────────────────────────────────────────
# Each entry: canonical id -> alternative surface forms AND paraphrases of the computation.
# Paraphrases matter because an NLA read usually describes rather than names: it says
# "repeatedly swapping adjacent elements", not "bubbleSort".
# Every pattern is anchored with \b on both ends unless it already ends in a suffix group.
# Two bugs this convention exists to prevent, both caught by the test suite: `prime` fired on
# "primer", and the bubble-sort paraphrase was written as the literal "swapping adjacent" so it
# missed the far more natural "swaps adjacent". Verb forms therefore go through `\w*`.
ALGORITHMS: Mapping[str, Sequence[str]] = {
    "bubble_sort": (r"\bbubble[- ]?sorts?\b", r"\bswap\w*\s+(the\s+)?(adjacent|neighbou?ring)\b",
                    r"\bexchang\w*\s+adjacent\b"),
    "quick_sort": (r"\bquick[- ]?sorts?\b", r"\bpivot\b", r"\bpartition\w*\s+around\b"),
    "merge_sort": (r"\bmerge[- ]?sorts?\b", r"\bmerging\s+sorted\b"),
    "insertion_sort": (r"\binsertion[- ]?sorts?\b", r"\binserts?\s+each\s+element\s+into\b"),
    "generic_sort": (r"\bsort(s|ing|ed)?\b", r"\bascending\s+order\b", r"\bdescending\s+order\b",
                     r"\borders?\s+the\s+(list|array|elements)\b"),
    "binary_search": (r"\bbinary\s+search\b", r"\bhalv(es|ing)\s+the\s+search\b",
                      r"\bmidpoint\s+of\s+the\s+range\b"),
    "linear_search": (r"\blinear\s+search\b", r"\bscans?\s+(the\s+)?(list|array)\s+for\b"),
    "fibonacci": (r"\bfib(onacci|fib)\b", r"\bsum\s+of\s+the\s+(two|three)\s+previous\b",
                  r"\beach\s+term\s+is\s+the\s+sum\s+of\b"),
    "factorial": (r"\bfactorials?\b", r"\bproduct\s+of\s+all\s+integers\s+up\s+to\b"),
    "gcd": (r"\bgreatest\s+common\s+divisor\b", r"\bgcd\b", r"\beuclid(ean)?\s+algorithm\b"),
    "prime_check": (r"\bprimes?\b", r"\bprimality\b", r"\bdivisible\s+by\s+any\b", r"\bsieve\b"),
    "palindrome": (r"\bpalindrom(e|es|ic)\b", r"\breads?\s+the\s+same\s+(forwards?|backwards?)\b",
                   r"\bsame\s+when\s+reversed\b"),
    "string_reverse": (r"\brevers(e|es|ing)\s+the\s+(string|characters|list|array)\b",),
    "binomial_coefficient": (r"\bbinomial\s+coefficients?\b", r"\bn\s+choose\s+k\b",
                             r"\bcombinations?\s+count\b", r"\bpascal'?s\s+triangle\b"),
    "matrix_multiply": (r"\bmatrix\s+multiplication\b", r"\bdot\s+product\s+of\s+rows\b"),
    "hashing": (r"\bhash(es|ing)?\b", r"\bhash\s+(function|table|map)\b", r"\bdigests?\b",
                r"\bchecksums?\b"),
    "graph_traversal": (r"\bbreadth[- ]first\b", r"\bdepth[- ]first\b", r"\bbfs\b", r"\bdfs\b",
                        r"\btraverses?\s+the\s+graph\b"),
    "dynamic_programming": (r"\bdynamic\s+programming\b", r"\bmemoiz\w*\b", r"\btabulation\b"),
    # NARROWED after validation on 5,090 banked reads. The original patterns included bare
    # `\bencod\w*\b` / `\bdecod\w*\b` / `\bciphers?\b`, which fired on reads *describing the
    # obfuscation* — "encoded variable names", "a cipher or log entry", "ASCII encoding
    # context". The tell was directional: `encoding` was the ONLY label whose rate rose with
    # obfuscation tier (L0 1.68% -> L1b 3.04% -> L3 3.69%) while every genuine content label
    # fell (generic_sort 8.9% -> 2.9%). A label that tracks the condition rather than the
    # content would have loaded straight onto the C1/C2 contrast it was meant to measure.
    # Encoding as a *computation* now requires an object, not a bare verb.
    "encoding": (r"\bbase64\b", r"\brot-?13\b", r"\bcaesar\s+cipher\b",
                 r"\b(en|de)cod(e|es|ing)\s+the\s+(string|text|message|input|data|bytes)\b",
                 r"\bencrypts?\s+the\s+(string|text|message|input|data)\b"),
    "counting": (r"\bcount(s|ing)?\s+the\s+(number|occurrences)\b", r"\bfrequency\s+of\b",
                 r"\btally\b"),
    "max_min": (r"\bmaximum\b", r"\blargest\s+value\b", r"\bminimum\b", r"\bsmallest\s+value\b"),
    "sum_accumulate": (r"\bsum(s|ming)?\s+(of|the)\b", r"\baccumulat\w*\b",
                       r"\brunning\s+total\b"),
}

ALGO_NONE = "ALGO_NONE"

_COMPILED: dict[str, re.Pattern] = {
    k: re.compile("|".join(f"(?:{p})" for p in pats), re.I) for k, pats in ALGORITHMS.items()
}

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Za-z])(?=[0-9])|_")


def split_identifiers(text: str) -> str:
    """Insert spaces at camelCase and snake_case boundaries.

    Reads quote source identifiers verbatim — the malware run produced reads naming
    `browser_cookie3` and `custom_function` straight out of the code — and a word-boundary
    pattern cannot see inside them: `\\bprime\\b` does not match `isPrime`, because `s` and `P`
    are both word characters. Scoring the split form as well as the original recovers those
    hits without loosening any pattern, which would cost precision everywhere else.
    """
    return _CAMEL.sub(" ", text)


def algorithm_claims(text: str) -> set[str]:
    """Every vocabulary algorithm the text asserts. May be empty; may be more than one.

    Matched against both the raw text and its identifier-split form, so `isPrime` and
    "checks whether a number is prime" both score as `prime_check`.
    """
    split = split_identifiers(text)
    return {k for k, pat in _COMPILED.items() if pat.search(text) or pat.search(split)}


def primary_claim(text: str) -> str:
    """Single best label: the algorithm with the most distinct pattern hits.

    Ties resolve to `ALGO_NONE` rather than to an arbitrary winner. A read that fires two
    algorithms equally is genuinely ambiguous, and forcing a pick would manufacture agreement.
    `generic_sort` loses to any specific sort so that "bubble sort" is not scored as a mere
    "sort" claim.
    """
    scored = text + "\n" + split_identifiers(text)
    hits = {k: len(set(_COMPILED[k].findall(scored))) for k in algorithm_claims(text)}
    if not hits:
        return ALGO_NONE
    specific = {k: v for k, v in hits.items() if k != "generic_sort"}
    pool = specific or hits
    top = max(pool.values())
    winners = [k for k, v in pool.items() if v == top]
    return winners[0] if len(winners) == 1 else ALGO_NONE


# ── Deception measurement ───────────────────────────────────────────────────
@dataclass(frozen=True)
class Read:
    """One NLA read, with what is needed to run every confound defence."""
    item_id: str
    position: int
    text: str
    condition: str                    # "C0" | "C1" | "C2"
    char_distance: float | None = None  # chars from the nearest misleading identifier


def deception_hit(read: Read, injected: str) -> bool:
    """Does this read assert the algorithm C2 injected?

    Membership, not the primary label: a read that names the injected algorithm alongside the
    true one has still been contaminated, and requiring it to *dominate* would understate the
    effect.
    """
    return injected in algorithm_claims(read.text)


def truth_hit(read: Read, true_algorithm: str) -> bool:
    return true_algorithm in algorithm_claims(read.text)


def recurrent_belief(reads: Sequence[Read], claim: str, k: int = 3,
                     min_gap: int = 2) -> bool:
    """Is `claim` asserted at >= k NON-ADJACENT positions?

    The NLA paper's own trust heuristic: claims recurring across token positions are more
    likely to reflect the represented content than a one-off. `min_gap` enforces
    non-adjacency, because two reads at neighbouring tokens see almost the same prefix and are
    not independent evidence.
    """
    hits = sorted(r.position for r in reads if claim in algorithm_claims(r.text))
    if len(hits) < k:
        return False
    chosen = [hits[0]]
    for p in hits[1:]:
        if p - chosen[-1] >= min_gap:
            chosen.append(p)
    return len(chosen) >= k


def distant_reads(reads: Sequence[Read], min_chars: float) -> list[Read]:
    """Reads far from the misleading identifier. Echo is local; a belief propagates."""
    return [r for r in reads if r.char_distance is not None and r.char_distance >= min_chars]


# ── Aggregates and the floor ────────────────────────────────────────────────
def item_deception_rate(reads_by_item: Mapping[str, Sequence[Read]], injected_by_item:
                        Mapping[str, str], k: int = 1) -> float:
    """Fraction of items where the injected algorithm is asserted.

    k=1 is "any read"; k>=3 applies the recurrence rule. Both are reported; the pre-registered
    primary is the recurrence version, because the any-read version is the one a lexical echo
    would inflate.
    """
    if not reads_by_item:
        return 0.0
    hits = 0
    for item, reads in reads_by_item.items():
        inj = injected_by_item.get(item)
        if inj is None:
            continue
        hits += int(recurrent_belief(reads, inj, k=k) if k > 1
                    else any(deception_hit(r, inj) for r in reads))
    return hits / len(reads_by_item)


def confabulation_floor(c1_reads_by_item: Mapping[str, Sequence[Read]],
                        injected_by_item: Mapping[str, str], k: int = 1) -> float:
    """The C1 rate: how often the *would-be* injected algorithm is asserted with no cue present.

    C1 is neutral renaming — the lexical content is destroyed but nothing false is supplied. Any
    C2 excess over this is the only part attributable to the injected name. Reporting a raw C2
    rate without it would be reporting confabulation.
    """
    return item_deception_rate(c1_reads_by_item, injected_by_item, k=k)


def behavioural_coupling(reads_by_item: Mapping[str, Sequence[Read]],
                         injected_by_item: Mapping[str, str],
                         correct_by_item: Mapping[str, bool], k: int = 1) -> dict[str, float]:
    """Deception rate split by run correctness.

    Defence 4: a constant lexical echo predicts NO difference between correct and incorrect
    runs. A belief that actually drives behaviour should be commoner where the run failed. This
    is the cheapest discriminator between the two accounts and needs no extra generation.
    """
    groups: dict[bool, dict[str, Sequence[Read]]] = {True: {}, False: {}}
    for item, reads in reads_by_item.items():
        if item in correct_by_item:
            groups[bool(correct_by_item[item])][item] = reads
    out = {
        "rate_correct": item_deception_rate(groups[True], injected_by_item, k=k),
        "rate_incorrect": item_deception_rate(groups[False], injected_by_item, k=k),
        "n_correct": float(len(groups[True])),
        "n_incorrect": float(len(groups[False])),
    }
    out["delta_incorrect_minus_correct"] = out["rate_incorrect"] - out["rate_correct"]
    return out


SUSPECT_RATIO = 1.5
SUSPECT_MIN_RATE = 0.01     # the elevated rate must be able to move a result at all
SUSPECT_MIN_COUNT = 20      # ...and rest on more than a handful of reads


def surface_tracking_report(reads_by_condition: Mapping[str, Sequence[str]],
                            baseline: str, ratio_threshold: float = SUSPECT_RATIO,
                            min_rate: float = SUSPECT_MIN_RATE,
                            min_count: int = SUSPECT_MIN_COUNT
                            ) -> dict[str, dict[str, float]]:
    """Per-label fire rate by condition, with the ratio against a baseline condition.

    THE DIAGNOSTIC THAT CAUGHT `encoding`. A label measuring what a program *computes* should
    fire no more often as the program gets harder to read — in the banked corpus every genuine
    content label falls with obfuscation tier (`generic_sort` 8.9% at L0 -> 2.9% at L3). A label
    whose rate *rises* is describing the obfuscation instead: `encoding` went 1.68% -> 3.69%
    because reads say "encoded variable names" about renamed identifiers.

    A label is flagged at `ratio > 1.5`, not `> 1`. A bare `> 1` test flagged `generic_sort` at
    1.01 and `fibonacci` at 1.07, and in every such case the driver was **L2**, where
    control-flow flattening genuinely introduces dispatch loops and accumulator structure — the
    banked probe reads at L2 dispatchers say "state machine", "while count < threshold". That is
    real content, not an artifact, so a hair-trigger threshold would condemn good labels.
    `encoding` sat at 2.2, well clear. Labels below `min_base_rate` on the baseline are ignored
    because their ratio is decided by a handful of reads.

    A flagged label must not be used in a contrast between those conditions; it loads on the
    condition by construction. Run this before freezing a vocabulary, not after reading a result.
    """
    rates: dict[str, dict[str, float]] = {}
    labels = set(ALGORITHMS) | {ALGO_NONE}
    per_cond: dict[str, Counter] = {}
    raw: dict[str, Counter] = {}
    for cond, texts in reads_by_condition.items():
        c = Counter(primary_claim(t) for t in texts)
        n = max(len(texts), 1)
        raw[cond] = c
        per_cond[cond] = Counter({k: c.get(k, 0) / n for k in labels})
    base = per_cond.get(baseline, Counter())
    others = [c for c in per_cond if c != baseline]
    for label in sorted(labels):
        row = {cond: round(per_cond[cond].get(label, 0.0), 5) for cond in per_cond}
        b = base.get(label, 0.0)
        worst = max((per_cond[c].get(label, 0.0) for c in others), default=0.0)
        worst_cond = max(others, key=lambda c: per_cond[c].get(label, 0.0)) if others else None
        worst_n = raw[worst_cond].get(label, 0) if worst_cond else 0
        row["worst_condition"] = worst_cond
        row["worst_n"] = float(worst_n)
        ratio = (worst / b) if b > 0 else (float("inf") if worst > 0 else 0.0)
        row["ratio_vs_baseline"] = round(ratio, 3)
        # Both floors apply to the ELEVATED cell, not the baseline. Guarding on the baseline
        # exempted the strongest possible signal — a label that never fires on clean code but
        # fires often under obfuscation has ratio = inf and baseline 0, and was silently waved
        # through. And a *rate* floor alone is not enough: `hashing` fired 4 times in 1,082 L2
        # reads against 0 at L0, which is 0.4% and ratio inf, but four reads cannot move any
        # result. A label is only worth flagging when it is both proportionally elevated and
        # carried by enough reads to matter — otherwise the gate cries wolf and gets ignored.
        row["suspect"] = float(ratio > ratio_threshold and worst >= min_rate
                               and worst_n >= min_count)
        rates[label] = row
    return rates


def gradeable_reads_per_item(reads_by_item: Mapping[str, Sequence[Read]]) -> dict[str, float]:
    """How many reads per item carry ANY algorithm claim — i.e. what k the design can afford.

    Measured on the banked corpus: median 2.0, mean 3.0, and only 49.7% of items reach 3. A
    pre-registered `k >= 3` recurrence rule would therefore be undefined or automatically
    negative for half the corpus — the same shape of failure as HT14, where 92% of cases never
    reached onset and the median was unmeasurable. Check this before freezing k.
    """
    counts = [sum(1 for r in reads if primary_claim(r.text) != ALGO_NONE)
              for reads in reads_by_item.values()]
    if not counts:
        return {"n_items": 0}
    counts.sort()
    n = len(counts)
    return {
        "n_items": n,
        "median": counts[n // 2],
        "mean": round(sum(counts) / n, 2),
        "frac_zero": round(sum(c == 0 for c in counts) / n, 4),
        "frac_ge_1": round(sum(c >= 1 for c in counts) / n, 4),
        "frac_ge_2": round(sum(c >= 2 for c in counts) / n, 4),
        "frac_ge_3": round(sum(c >= 3 for c in counts) / n, 4),
    }


def vocabulary_coverage(texts: Iterable[str]) -> dict[str, float | Counter]:
    """Diagnostic: how often the vocabulary fires at all, and how concentrated it is.

    A vocabulary that fires on 5% of reads is measuring nothing; one where a single label is
    90% of fires is degenerate. N7's kappa failed because the second rater answered one label
    91% of the time, and that is exactly the failure this surfaces — before the run, not after.
    """
    texts = list(texts)
    fired = [primary_claim(t) for t in texts]
    counts = Counter(f for f in fired if f != ALGO_NONE)
    n_fired = sum(counts.values())
    return {
        "n_texts": len(texts),
        "fire_rate": n_fired / max(len(texts), 1),
        "n_distinct_labels": len(counts),
        "max_label_share": (max(counts.values()) / n_fired) if n_fired else 0.0,
        "counts": counts,
    }
