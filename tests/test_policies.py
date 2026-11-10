"""CPU check: PolicyKVCache (streaming) sees exactly the frames attention_mask() allows."""
import torch
from fdkv.policies import Policy, attention_mask, PolicyKVCache

class Ref: keep = None

def visible_sets_streaming(p, keep):
    ref = Ref(); c = PolicyKVCache(1, 1, 4, p, ref, 'cpu', torch.float32)
    out = []
    for s in range(len(keep)):
        ref.keep = keep[s:s+1]
        r = c.complete(torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4), None)
        out.append(set(r.positions[0][r.positions[0] >= 0].tolist()))
    return out

torch.manual_seed(0)
for trial in range(30):
    S = 200
    keep = torch.rand(S) < torch.rand(1).item()
    keep[0] = True
    for p in [Policy('B0', 17), Policy('B1', 17, n_sink=4), Policy('P1', 17, skip_silence=True),
              Policy("P1", 9, n_sink=2, skip_silence=True), Policy("B0", 3),
              Policy("P1W", 17, n_sink=4, skip_silence=True, recent=5), Policy("P1W", 12, skip_silence=True, recent=11),
              Policy("B1", 17, n_sink=4, recent=6)]:
        m = attention_mask(p, keep)
        st = visible_sets_streaming(p, keep)
        for s in range(S):
            assert set(torch.nonzero(m[s]).flatten().tolist()) == st[s], (p, s)
        assert m.sum(1).max() <= p.slots
# P1 with keep all == B0
keep = torch.ones(300, dtype=torch.bool)
assert torch.equal(attention_mask(Policy('P1', 50, skip_silence=True), keep), attention_mask(Policy('B0', 50), keep))
print("OK: streaming cache == mask for all policies; P1(no silence) == B0")
