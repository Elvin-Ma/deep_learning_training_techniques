import math
import torch
from torch import nn

class T5RelativePositionBias(nn.Module):
    def __init__(self, num_heads, relative_attention_num_buckets=32):
        super().__init__()
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        
        # 定义可学习的相对位置偏置参数
        self.relative_attention_bias = nn.Embedding(
            relative_attention_num_buckets, num_heads
        )

    def _relative_position_bucket(self, relative_position):
        """
        将相对位置映射到离散的桶(bucket)
        """
        num_buckets = self.relative_attention_num_buckets
        ret = 0
        
        # 处理正向和负向相对位置
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))
        
        # 分桶策略
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / 
            torch.log(torch.tensor(num_buckets / max_exact)) * 
            (num_buckets - max_exact)
        ).to(torch.long)
        
        val_if_large = torch.min(
            val_if_large, 
            torch.full_like(val_if_large, num_buckets - 1)
        )
        
        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, query_len, key_len):
        """
        生成相对位置偏置矩阵
        Args:
            query_len: 查询序列长度
            key_len: 键序列长度
        Returns:
            bias: [num_heads, query_len, key_len]
        """
        # 生成相对位置矩阵
        context_position = torch.arange(query_len)[:, None]
        memory_position = torch.arange(key_len)[None, :]
        relative_position = memory_position - context_position
        
        # 映射到桶索引
        rp_bucket = self._relative_position_bucket(relative_position)
        
        # 查表获取偏置值
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1])  # [heads, q_len, k_len]
        return values

class T5Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # 初始化相对位置编码模块
        self.relative_position = T5RelativePositionBias(num_heads)
        
        # 初始化Q/K/V投影层
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 计算Q/K/V
        q = self.q(hidden_states)  # [batch, seq, d_model]
        k = self.k(hidden_states)
        v = self.v(hidden_states)
        
        # 拆分多头
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        
        # 添加相对位置偏置
        rel_pos_bias = self.relative_position(seq_len, seq_len)
        scores += rel_pos_bias
        
        # 计算注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 应用注意力到V
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return context
    
if __name__ == '__main__':
    # 创建一个T5Attention实例
    attention = T5Attention(d_model=512, num_heads=8)
    
    # 假设输入是一个[batch, seq, d_model]的tensor
    input_tensor = torch.randn(1, 32, 512)
    
    # 应用T5Attention
    output = attention(input_tensor)
    
    print(output.shape)