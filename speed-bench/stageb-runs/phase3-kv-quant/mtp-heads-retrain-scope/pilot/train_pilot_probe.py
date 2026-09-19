#!/usr/bin/env python3
"""
Tier-A MTP predictability probe (see PILOT-plan.md).

Answers, cheaply: from the FROZEN base's last hidden state at position t (plus the
embedding of the committed next token t+1), how accurately can a small head predict
token t+2 (and t+3) on real text? This is the ceiling every drafter lives under; the
go/no-go gate on t+2 top-1 accuracy decides whether the heavier native-head pilot
(Tier-B) is worth it.

Base: Qwen/Qwen3.8-Flash-Next (HF, full precision). REQUIRES a GPU + the base download;
it does NOT run on the M5 Pro (we hold only the 2-bit GGUF, which does not expose hidden
states). This script is prepared in-house and is UNTESTED here — validate on the GPU box.

Two stages:
  precompute : run the base once over a corpus, dump (h_t, id_{t+1}, id_{t+2}, id_{t+3})
  fit        : fit small t+2 / t+3 heads on the dump, print top-1 accuracy + the gate

Usage:
  python train_pilot_probe.py precompute --base Qwen/Qwen3.8-Flash-Next \
      --corpus corpus.txt --out dump.pt --max-tokens 3000000 --dtype int8
  python train_pilot_probe.py fit --dump dump.pt --epochs 3
"""
import argparse, sys, math

# ----- Tier-A gate (from PILOT-plan.md) -----
GATE_STRONG, GATE_MARGINAL = 0.70, 0.55


def stage_precompute(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.base, trust_remote_code=True)
    kw = dict(trust_remote_code=True, output_hidden_states=True, device_map="auto")
    if a.dtype == "int8":   kw.update(load_in_8bit=True)
    elif a.dtype == "fp8":  kw.update(torch_dtype=torch.float8_e4m3fn)
    else:                    kw.update(torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(a.base, **kw).eval()
    emb = model.get_input_embeddings().weight            # [V, H], tie to output space
    ids = tok(open(a.corpus).read(), return_tensors="pt").input_ids[0][: a.max_tokens]
    H = emb.shape[1]
    hs, e1, y2, y3 = [], [], [], []
    win = a.window
    with torch.no_grad():
        for s in range(0, len(ids) - 3, win):
            chunk = ids[s : s + win].unsqueeze(0).to(model.device)
            out = model(chunk)
            h = out.hidden_states[-1][0].to("cpu")        # [T, H] last hidden per pos
            # position t predicts t+2; head also sees embedding of the committed t+1
            T = h.shape[0]
            for t in range(T - 2):
                gi = s + t
                if gi + 3 >= len(ids): break
                hs.append(h[t]); e1.append(ids[gi + 1])
                y2.append(ids[gi + 2]); y3.append(ids[gi + 3])
            print(f"\r precompute {s+win}/{len(ids)}", end="", file=sys.stderr)
    torch.save(dict(H=H, hidden=torch.stack(hs), emb_id=torch.tensor(e1),
                    y2=torch.tensor(y2), y3=torch.tensor(y3),
                    emb_table=emb.detach().to("cpu").float()), a.out)
    print(f"\n wrote {a.out}: {len(hs)} samples, H={H}", file=sys.stderr)


class ProbeHead(__import__("torch").nn.Module):
    """RMSNorm over [h_t, emb(id_{t+1})] -> 2-layer MLP -> logits over the tied vocab.
    Mirrors the native nextn adapter (enorm/hnorm + eh_proj) at a light scale."""
    def __init__(self, H, V, emb_table, hidden=4096):
        import torch, torch.nn as nn
        super().__init__()
        self.nh = nn.RMSNorm(H); self.ne = nn.RMSNorm(H)
        self.eh = nn.Linear(2 * H, H, bias=False)          # eh_proj analogue
        self.mlp = nn.Sequential(nn.Linear(H, hidden), nn.SiLU(), nn.Linear(hidden, H))
        self.out = nn.Linear(H, V, bias=False)
        with torch.no_grad(): self.out.weight.copy_(emb_table)   # tie to base embeddings
        self.out.weight.requires_grad_(False)

    def forward(self, h, e):
        import torch
        x = self.eh(torch.cat([self.nh(h), self.ne(e)], -1))
        return self.out(x + self.mlp(x))


def stage_fit(a):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    d = torch.load(a.dump)
    H, V = d["H"], d["emb_table"].shape[0]
    emb_table, dev = d["emb_table"], ("cuda" if torch.cuda.is_available() else "cpu")
    E = emb_table[d["emb_id"]]                              # emb of committed t+1
    n = len(d["hidden"]); ntr = int(n * 0.9)
    for depth, y in (("t+2", d["y2"]), ("t+3", d["y3"])):
        head = ProbeHead(H, V, emb_table).to(dev)
        opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=a.lr)
        tr = DataLoader(TensorDataset(d["hidden"][:ntr], E[:ntr], y[:ntr]),
                        batch_size=a.bs, shuffle=True)
        for ep in range(a.epochs):
            head.train()
            for h, e, t in tr:
                h, e, t = h.to(dev), e.to(dev), t.to(dev)
                loss = torch.nn.functional.cross_entropy(head(h, e), t)
                opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            hv, ev, tv = d["hidden"][ntr:].to(dev), E[ntr:].to(dev), y[ntr:].to(dev)
            top1 = (head(hv, ev).argmax(-1) == tv).float().mean().item()
        verdict = ("STRONG -> proceed to Tier-B" if top1 >= GATE_STRONG else
                   "MARGINAL -> Tier-B with eyes open" if top1 >= GATE_MARGINAL else
                   "LOW -> STOP, keep ~43 t/s + multi-session")
        print(f"[{depth}] top-1 acc = {top1*100:.1f}%  ->  {verdict}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)
    pc = sub.add_parser("precompute"); pc.add_argument("--base", required=True)
    pc.add_argument("--corpus", required=True); pc.add_argument("--out", default="dump.pt")
    pc.add_argument("--max-tokens", type=int, default=3_000_000)
    pc.add_argument("--window", type=int, default=4096)
    pc.add_argument("--dtype", choices=["int8", "fp8", "bf16"], default="int8")
    ft = sub.add_parser("fit"); ft.add_argument("--dump", default="dump.pt")
    ft.add_argument("--epochs", type=int, default=3); ft.add_argument("--bs", type=int, default=256)
    ft.add_argument("--lr", type=float, default=1e-3)
    a = p.parse_args()
    {"precompute": stage_precompute, "fit": stage_fit}[a.stage](a)
