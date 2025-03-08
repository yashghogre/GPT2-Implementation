import torch
import torch.nn as nn
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


embd = Embedding()
x_emb = embd(enc_inp)
pos_e = PositionalEncoding()
x_pos = pos_e(x_emb)
