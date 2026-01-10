import torch
import torch.nn as nn
import math

class CrossAttention(nn.Module):
    def __init__(self, dimension_q, dimension_kv, num_head, d_k, d_v, d_o, dropout=0.1):
        """
        dimension_q  : 输入 Query 的维度（通常是 decoder 的 d_model）
        dimension_kv : 输入 Key/Value 的维度（通常是 encoder 的 d_model；也可以不同）
        num_head     : 头数 h
        d_k          : 每个 head 的 Q/K 维度
        d_v          : 每个 head 的 V 维度
        d_o          : 输出维度（通常设成 decoder 的 d_model）
        """
        super().__init__()
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_o = d_o

        # 1) 线性投影：Q 来自 decoder，K/V 来自 encoder
        self.fc_q = nn.Linear(dimension_q,  num_head * d_k)
        self.fc_k = nn.Linear(dimension_kv, num_head * d_k)
        self.fc_v = nn.Linear(dimension_kv, num_head * d_v)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # 2) 输出投影：把 h*d_v 映射回 d_o
        self.fc_o = nn.Linear(num_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):
        """
        q: (b, n_q, dimension_q)    -> decoder hidden states
        k: (b, n_k, dimension_kv)   -> encoder outputs
        v: (b, n_k, dimension_kv)   -> encoder outputs (通常 v 和 k 来自同一个张量)
        mask: (b, n_q, n_k) 或 (b, 1, 1, n_k) 等可广播形状
              - padding mask：把 encoder padding 位置设为 -inf
              - 一般 cross-attn 不需要因果 mask（因果 mask 发生在 decoder self-attn）
        """
        b, n_q, _ = q.size()
        _, n_k, _ = k.size()
        _, n_v, _ = v.size()
        assert n_v == n_k, "Cross-attention: usually n_v should equal n_k."

        # 1) 线性映射到多头空间
        q_proj = self.fc_q(q)  # (b, n_q, h*d_k)
        k_proj = self.fc_k(k)  # (b, n_k, h*d_k)
        v_proj = self.fc_v(v)  # (b, n_k, h*d_v)

        # 2) 拆分多头：(b, n, h*d) -> (b, h, n, d)
        Q = q_proj.view(b, n_q, self.num_head, self.d_k).transpose(1, 2)  # (b, h, n_q, d_k)
        K = k_proj.view(b, n_k, self.num_head, self.d_k).transpose(1, 2)  # (b, h, n_k, d_k)
        V = v_proj.view(b, n_k, self.num_head, self.d_v).transpose(1, 2)  # (b, h, n_k, d_v)

        # 3) 注意力分数：QK^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        # scores: (b, h, n_q, n_k)

        # 4) mask（如果传入的是 (b, n_q, n_k)，就扩展到 head 维度）
        if mask is not None:
            # 常见 mask: (b, n_q, n_k) -> (b, 1, n_q, n_k)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores + mask

        # 5) softmax 得到 attention 权重
        attn = self.softmax(scores)        # (b, h, n_q, n_k)
        attn = self.dropout(attn)

        # 6) 加权求和：attn @ V
        head_out = torch.matmul(attn, V)   # (b, h, n_q, d_v)

        # 7) 合并多头：(b, h, n_q, d_v) -> (b, n_q, h*d_v)
        head_out = head_out.transpose(1, 2).contiguous().view(b, n_q, self.num_head * self.d_v)

        # 8) 输出投影回 d_o
        out = self.fc_o(head_out)          # (b, n_q, d_o)

        return attn, out


if __name__ == "__main__":
    # ----------- 测试一个典型 cross-attn 场景 -----------
    b = 2
    n_q = 5   # decoder 当前长度
    n_k = 7   # encoder 源序列长度

    d_dec = 64   # dimension_q
    d_enc = 128  # dimension_kv

    h = 4
    d_k = 16
    d_v = 16
    d_o = 64     # 通常输出回 decoder 的 d_model

    q = torch.randn(b, n_q, d_dec)   # decoder states
    k = torch.randn(b, n_k, d_enc)   # encoder outputs
    v = torch.randn(b, n_k, d_enc)   # encoder outputs

    # padding mask 示例：假设 encoder 最后两个位置是 padding
    # mask 中 padding 位置给 -inf，其他位置为 0
    mask = torch.zeros(b, n_q, n_k)
    mask[:, :, -2:] = float("-inf")

    ca = CrossAttention(dimension_q=d_dec, dimension_kv=d_enc, num_head=h, d_k=d_k, d_v=d_v, d_o=d_o)
    attn, out = ca(q, k, v, mask)

    print("attn:", attn.shape)  # (b, h, n_q, n_k)
    print("out :", out.shape)   # (b, n_q, d_o)
