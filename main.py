import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer
import math

GPT2_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

print(f"device:\t{GPT2_CONFIG['device']}")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "My name is Yash"
enc_inp = tokenizer.encode(text, return_tensors='pt').to(GPT2_CONFIG["device"])
print(f"input tokens shape:\t{enc_inp.shape}")

class Embedding(nn.Module):
    def __init__(self, vocab_size=GPT2_CONFIG["vocab_size"], emb_dim=GPT2_CONFIG["emb_dim"], device=GPT2_CONFIG["device"]):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, device=device)

    def forward(self, x):
        out = self.emb(x)

        print(f"embedded tokens shape:\t{out.shape}")
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len=GPT2_CONFIG["context_length"], emb_dim=GPT2_CONFIG["emb_dim"], device=GPT2_CONFIG["device"]):
        super().__init__()

        pe = torch.zeros(max_seq_len, emb_dim, device=device)
        position = torch.arange(0, max_seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * (-math.log(10000.0) / emb_dim)).to(device)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        out = x + self.pe[:x.size(1), :]

        assert x.shape == out.shape, "input embedding shape does not match positional encoding shape"
        print(f"positional encoding shape:\t{out.shape}")
        return out

class Head(nn.Module):
    def __init__(self, head_size, emb_dim=GPT2_CONFIG["emb_dim"], qkv_bias=GPT2_CONFIG["qkv_bias"], device=GPT2_CONFIG["device"]):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.head_size = head_size

        self.qw = nn.Linear(emb_dim, head_size, bias=qkv_bias, device=device)
        self.kw = nn.Linear(emb_dim, head_size, bias=qkv_bias, device=device)
        self.vw = nn.Linear(emb_dim, head_size, bias=qkv_bias, device=device)

    def forward(self, x, device=GPT2_CONFIG["device"]):
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        mask = torch.triu(torch.full((x.size(1), x.size(1)), float('-inf'), device=device), diagonal=1)

        qk = q @ k.transpose(-2, -1)
        scaling = qk * (self.emb_dim ** -0.5)
        add_mask = scaling + mask
        scaled_sm = F.softmax(add_mask, dim=-1)
        qk_v = scaled_sm @ v

        print(f"head shape:\t{qk_v.shape}")
        return qk_v

class Multi_Head(nn.Module):
    def __init__(self, emb_dim=GPT2_CONFIG["emb_dim"], n_heads=GPT2_CONFIG["n_heads"], device=GPT2_CONFIG["device"]):
        super().__init__()

        self.head_size = emb_dim // n_heads

        self.heads = nn.ModuleList([Head(self.head_size) for _ in range(n_heads)])
        self.lyr = nn.Linear(emb_dim, emb_dim, bias=False, device=device)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.lyr(out)

        assert x.shape == out.shape, "positional encoding input shape does not match multi-head output shape"
        print(f"multi head shape: {out.shape}")
        return out


embd = Embedding()
x_emb = embd(enc_inp)
pos_e = PositionalEncoding()
x_pos = pos_e(x_emb)
# head = Head(head_size=GPT2_CONFIG["emb_dim"])
# x_head = head(x_pos)
m_head = Multi_Head()
x_m_head = m_head(x_pos)
