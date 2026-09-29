import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from sf2.data import lora  # noqa: E402


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8)
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(8, 8)
        self.other = nn.Linear(8, 8)

    def forward(self, x):
        return self.mlp.down_proj(self.self_attn.q_proj(x)) + self.other(x)


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = nn.Module()
        self.text_model.layers = nn.ModuleList([Layer(), Layer()])
        self.text_model.norm = nn.LayerNorm(8)

    def forward(self, x):
        for l in self.text_model.layers:
            x = l(x)
        return x


def test_inject_starts_as_identity_and_merges_exactly(monkeypatch):
    monkeypatch.setattr(lora, "text_layers", lambda enc: enc.text_model.layers)
    torch.manual_seed(0)
    enc = Enc()
    x = torch.randn(3, 8)
    y0 = enc(x)
    assert lora.inject(enc, rank=4, dropout=0.0) == 4
    assert torch.allclose(enc(x), y0)  # B starts at zero
    with torch.no_grad():
        for p in lora.lora_params(enc):
            p.add_(torch.randn_like(p) * 0.1)
    y1 = enc(x)
    merged = lora.merge(enc)
    assert not lora.lora_params(merged)
    assert torch.allclose(merged(x), y1, atol=1e-5)
    assert set(merged.state_dict()) == set(Enc().state_dict())  # plain checkpoint keys
