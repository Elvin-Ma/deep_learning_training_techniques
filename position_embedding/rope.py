import torch
import torch.nn as nn
import math

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        seq = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', seq, self.inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # Convert to complex numbers
        return freqs_cis

def apply_rotary_pos_emb(q, k, freqs_cis):
    q_embed = q * freqs_cis
    k_embed = k * freqs_cis
    return q_embed, k_embed

# 示例使用
if __name__ == "__main__":
    dim = 64  # 位置编码的维度
    max_seq_len = 2048  # 最大序列长度
    seq_len = 128  # 当前序列长度

    rotary_emb = RotaryPositionEmbedding(dim, max_seq_len)
    freqs_cis = rotary_emb(seq_len, device='cpu')

    # 假设 q 和 k 是来自 Transformer 的查询和键
    q = torch.randn(seq_len, dim // 2, 2)  # 实部和虚部
    k = torch.randn(seq_len, dim // 2, 2)  # 实部和虚部

    # 将 q 和 k 转换为复数
    q_complex = torch.complex(q[..., 0], q[..., 1])
    k_complex = torch.complex(k[..., 0], k[..., 1])

    q_embed_complex, k_embed_complex = apply_rotary_pos_emb(q_complex, k_complex, freqs_cis)

    # 将复数结果转换回实部和虚部
    q_embed = torch.stack((q_embed_complex.real, q_embed_complex.imag), dim=-1)
    k_embed = torch.stack((k_embed_complex.real, k_embed_complex.imag), dim=-1)

    print("Query with Rotary Position Embedding (Real):\n", q_embed[..., 0])
    print("Query with Rotary Position Embedding (Imag):\n", q_embed[..., 1])
    print("Key with Rotary Position Embedding (Real):\n", k_embed[..., 0])
    print("Key with Rotary Position Embedding (Imag):\n", k_embed[..., 1])