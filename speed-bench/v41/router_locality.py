#!/usr/bin/env python3
"""Routing locality from a DS4_V41_ROUTER_LOG file (docs/V41_64GB_BUILD.md §7 step 4).

LRU expert-cache hit rate versus budget, and how much of the next token's
experts the previous tokens already selected (the ceiling for prefetch).
"""
import argparse
import collections
import json
import statistics
import sys

GIB = 1024 ** 3


def parse_router_log(lines):
    """tokens[t][layer] -> tuple of expert ids, in log order."""
    tokens, current, cur_pos = [], [], None
    for raw in lines:
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = [int(x) for x in raw.split()]
        pos, layer, ids = fields[0], fields[1], tuple(fields[2:])
        if pos != cur_pos:
            if current:
                tokens.append(current)
            current, cur_pos = [], pos
        if layer != len(current):
            raise ValueError(f"pos {pos}: layer {layer} out of order")
        current.append(ids)
    if current:
        tokens.append(current)
    if tokens and any(len(t) != len(tokens[0]) for t in tokens):
        raise ValueError("tokens have different layer counts")
    return tokens


def lru_hit_rates(tokens, unit_bytes, budgets, warmup=0):
    """budget bytes -> hit rate of a global LRU over (layer, expert), after `warmup` tokens."""
    out = {}
    for budget in budgets:
        cache, used, hits, total = collections.OrderedDict(), 0, 0, 0
        for t, layers in enumerate(tokens):
            for layer, ids in enumerate(layers):
                for expert in ids:
                    key = (layer, expert)
                    hit = key in cache
                    if hit:
                        cache.move_to_end(key)
                    else:
                        cache[key] = unit_bytes[layer]
                        used += unit_bytes[layer]
                        while used > budget and cache:
                            used -= cache.popitem(last=False)[1]
                    if t >= warmup:
                        total += 1
                        hits += hit
        out[budget] = hits / total if total else 0.0
    return out


def overlap_stats(tokens, ks=(1, 2, 4, 8)):
    """Same-layer overlap with the previous token, and coverage by the union of the last K."""
    pair, cover = [], {k: [] for k in ks}
    n_layer = len(tokens[0]) if tokens else 0
    for t in range(1, len(tokens)):
        for layer in range(n_layer):
            now = set(tokens[t][layer])
            pair.append(len(now & set(tokens[t - 1][layer])) / len(now))
            for k in ks:
                seen = set()
                for back in tokens[max(0, t - k):t]:
                    seen.update(back[layer])
                cover[k].append(len(now & seen) / len(now))

    def mean(xs):
        return statistics.fmean(xs) if xs else 0.0

    return {"pair_overlap": mean(pair), "union_cover": {str(k): mean(v) for k, v in cover.items()}}


def new_experts_per_token(tokens):
    """Mean number of never-seen (layer, expert) pairs per token over the second half."""
    seen, fresh = set(), []
    for layers in tokens:
        n = 0
        for layer, ids in enumerate(layers):
            for expert in ids:
                if (layer, expert) not in seen:
                    seen.add((layer, expert))
                    n += 1
        fresh.append(n)
    tail = fresh[len(fresh) // 2:]
    return statistics.fmean(tail) if tail else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log")
    ap.add_argument("--bytes", required=True, help="gguf_bytes.py JSON (expert_unit_bytes)")
    ap.add_argument("--budgets-gib", default="2,4,8,12,16,24,32,48")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--out")
    args = ap.parse_args()
    with open(args.log) as fp:
        tokens = parse_router_log(fp)
    if not tokens:
        raise SystemExit("router_locality: empty log")
    with open(args.bytes) as fp:
        units = json.load(fp)["expert_unit_bytes"]
    unit_bytes = [units[str(layer)] for layer in range(len(tokens[0]))]
    budgets = [float(x) * GIB for x in args.budgets_gib.split(",")]
    rates = lru_hit_rates(tokens, unit_bytes, budgets, args.warmup)
    result = {"tokens": len(tokens), "layers": len(tokens[0]), "warmup": args.warmup,
              "lru_hit": {f"{b / GIB:g}": r for b, r in rates.items()},
              "new_per_token": new_experts_per_token(tokens), **overlap_stats(tokens)}
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as fp:
            fp.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
