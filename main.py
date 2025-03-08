''' Imported Libraries '''

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2Tokenizer
import math

''' Config '''

GPT2_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_blocks": 12,
        "drop_rate": 0.1,
        "qkv_bias": False,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
        }

GPT2_CONFIG["ff_hidden_size"] = 4 * GPT2_CONFIG["emb_dim"]

print(f"device:\t{GPT2_CONFIG['device']}")

''' tokenizer '''

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "My name is Yash"
enc_inp = tokenizer.encode(text, return_tensors='pt').to(GPT2_CONFIG["device"])
print(f"input tokens shape:\t{enc_inp.shape}")

''' Embedding '''

class Embedding(nn.Module):
    def __init__(self, vocab_size=GPT2_CONFIG["vocab_size"], emb_dim=GPT2_CONFIG["emb_dim"], device=GPT2_CONFIG["device"]):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, device=device)

    def forward(self, x):
        out = self.emb(x)

        print(f"embedded tokens shape:\t{out.shape}")
        return out

''' Positional Encoding '''

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

''' Masked Attention Head '''

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

''' Multi Head Masked Attention '''

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
        print(f"multi head shape:\t{out.shape}")
        return out

''' Feed Forward Layer '''

class FeedForward(nn.Module):
    def __init__(self, emb_dim=GPT2_CONFIG["emb_dim"], ff_hidden_size=GPT2_CONFIG["ff_hidden_size"], device=GPT2_CONFIG["device"]):
        super().__init__()

        self.lyr_1 = nn.Linear(emb_dim, ff_hidden_size, device=device)
        self.lyr_2 = nn.Linear(ff_hidden_size, emb_dim, device=device)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.lyr_1(x)
        out = self.gelu(out)
        out = self.lyr_2(out)

        print(f"feed forward shape:\t{out.shape}")
        return out

''' Transformer Block '''

class Block(nn.Module):
    def __init__(self, emb_dim=GPT2_CONFIG["emb_dim"], drop_rate=GPT2_CONFIG["drop_rate"]):
        super().__init__()

        self.m_head = Multi_Head()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        out = self.ln1(x + self.dropout(self.m_head(x)))
        out = self.ln2(out + self.dropout(self.ff(out)))

        print(f"block shape:\t{out.shape}")
        return out

''' GPT2 Model Class '''

class GPT2(nn.Module):
    def __init__(self, n_blocks=GPT2_CONFIG["n_blocks"], emb_dim=GPT2_CONFIG["emb_dim"], vocab_size=GPT2_CONFIG["vocab_size"], device=GPT2_CONFIG["device"]):
        super().__init__()

        self.embd = Embedding()
        self.pos_e = PositionalEncoding()

        self.block_list = nn.ModuleList([Block() for _ in range(n_blocks)])
        self.lyr = nn.Linear(emb_dim, vocab_size, device=device)

    def forward(self, x):
        out = self.pos_e(self.embd(x))

        for block in self.block_list:
            out = block(out)

        out = self.lyr(out)

        print(f"GPT2 shape:\t{out.shape}")
        return out


gpt2 = GPT2()
gpt2.to(GPT2_CONFIG["device"])
x_gpt2 = gpt2(enc_inp)

gpt2_params = sum(p.numel() for p in gpt2.parameters())
print(f"GPT2 Params:\t{gpt2_params}")
