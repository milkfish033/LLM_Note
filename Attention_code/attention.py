import torch
import torch.nn as nn
import math

class Attention(nn.Module):
    def __init__(self, dimension_k, dimension_v, d_k, d_v, d_o, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_o = d_o

        # 线性投影到注意力空间（单头：输出维度就是 d_k || d_v）
        self.fc_q = nn.Linear(dimension_k, d_k) #Q
        self.fc_k = nn.Linear(dimension_k, d_k) #K
        self.fc_v = nn.Linear(dimension_v, d_v) #V

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        # 输出投影
        self.fc_o = nn.Linear(d_v, d_o)

    def forward(self, q, k, v, mask=None):
        """
        q: (b, n_q, dimension_k)
        k: (b, n_k, dimension_k)
        v: (b, n_v, dimension_v)  通常 n_v == n_k
        mask: (b, n_q, n_k)  允许为 None
        b: bacth size
        n_q: query 序列长度
        n_k: key 序列长度
        n_v: value 序列长度
        """
        b, n_q, _ = q.size()
        _, n_k, _ = k.size()
        _, n_v, _ = v.size()

        # 1) 线性映射
        Q = self.fc_q(q)  # (b, n_q, d_k)
        K = self.fc_k(k)  # (b, n_k, d_k)
        V = self.fc_v(v)  # (b, n_v, d_v)

        # 2) 注意力分数 scores = QK^T / sqrt(d_k)
        #    (b, n_q, d_k) @ (b, d_k, n_k) -> (b, n_q, n_k)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)

        # 3) 加 mask（比如因果 mask 或 padding mask）
        if mask is not None:
            # mask 需要是 (b, n_q, n_k)，mask 位置为 -inf
            scores = scores + mask

        # 4) softmax -> attention weights
        attn = self.softmax(scores)      # (b, n_q, n_k)
        attn = self.dropout(attn)

        # 5) 加权求和：attn @ V
        #    (b, n_q, n_k) @ (b, n_k, d_v) -> (b, n_q, d_v)
        out = torch.matmul(attn, V)

        # 6) 输出投影
        out = self.fc_o(out)  # (b, n_q, d_o)

        return attn, out


if __name__ == "__main__":
    # ----------- 测试 -----------
    batch = 10
    n_q, n_k, n_v = 4, 4, 4
    dimension_k = 128
    dimension_v = 64
    d_k, d_v, d_o = 16, 16, 8

    #如果自注意力，则 q = k = v = x
    q = torch.randn(batch, n_q, dimension_k)
    k = torch.randn(batch, n_k, dimension_k)
    v = torch.randn(batch, n_v, dimension_v)

    # 构造一个上三角因果 mask（对未来位置置 -inf）
    mask = torch.full((batch, n_q, n_k), float("-inf"))
    mask = torch.triu(mask, diagonal=1)

    attn = Attention(dimension_k, dimension_v, d_k, d_v, d_o)
    attention, output = attn(q, k, v, mask)

    print(attention.size(), output.size())
    # attention: (b, n_q, n_k)
    # output:    (b, n_q, d_o)
