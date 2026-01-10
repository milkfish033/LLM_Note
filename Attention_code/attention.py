import math
import torch 
'function to compute attention matrix'
def attention(query, key, value, dropout = None):
    """
    query: [batch_size, heads, seq_q, d_k]
    key: [batch_size, heads, seq_q, d_k]
    value: [batch_size, heads, seq_q, d_k]
    bacth_size: 一次并行送进模型、一起计算的样本数量
    heads: 同一层里的「多头注意力」数量
    seq_q: query token 的数量(query 序列长度)
    dropout:正则化, 是为了防止 softmax 产生“过度集中的注意力, 通过随机断开部分注意力连接，迫使模型学习更稳健、可泛化的对齐模式
    """
    d_k = query.size(-1)
    #计算Q*K^T
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    #softmax 
    p_attn = scores.softmax(dim = -1) #对于每一个 query，我在所有 key 之间分配注意力权重; softmax(dim = key 这一维)  → dim = -1
    #正则化，防止过拟合
    if dropout is not None:
        p_attn = dropout(p_attn)
    #multiply with value
    return torch.matmul(p_attn, value), p_attn


#self-attention
query = x * W_Q
key = x * W_K
value = x * W_V
out, attn = attention(query, key, value, None)


def mask_attention(query, key, value, dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 创建一个上三角矩阵，用于遮蔽未来信息。
    # 先通过 full 函数创建一个 1 * seq_len * seq_len 的矩阵
    mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
    # triu 函数的功能是创建一个上三角矩阵
    mask = torch.triu(mask, diagonal=1)
    # 此处的 scores 为计算得到的注意力分数，mask 为上文生成的掩码矩阵
    scores = scores + mask[:, :seqlen, :seqlen]
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    
    