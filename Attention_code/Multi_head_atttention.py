import torch
import torch.nn as nn
import math

class MHA(nn.Module):
    def __init__(self, num_head, dimension_k, dimension_v, d_k, d_v, d_o, dropout=0.1):
        super().__init__()
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_o = d_o
        
        # 线性投影到多头空间 (Linear projection to multi-head space)
        self.fc_q = nn.Linear(dimension_k, num_head * d_k)
        self.fc_k = nn.Linear(dimension_k, num_head * d_k)
        self.fc_v = nn.Linear(dimension_v, num_head * d_v)
        
        self.dropout = nn.Dropout(dropout)
        self.fc_o = nn.Linear(num_head * d_v, d_o)
        self.softmax = nn.Softmax(dim=-1) # 对最后一维(n_k)做softmax

    def forward(self, q, k, v, mask):
        # q:(b, n_q, dimension_k)
        # k:(b, n_k, dimension_k)
        # v:(b, n_v, dimension_v)
        
        b, n_q, _ = q.size()
        _, n_k, _ = k.size()
        _, n_v, _ = v.size()
        
        h = self.num_head
        
        # 线性映射 (Linear mapping)
        # Fixed typo: '9' -> 'Q', 'g' -> 'q'
        Q = self.fc_q(q) # (b, n_q, h*d_k)
        K = self.fc_k(k) # (b, n_k, h*d_k)
        V = self.fc_v(v) # (b, n_v, h*d_v)
        
        # 变形+维度对齐到多头 (Reshape + align to multi-head): (b, h, n_*, d_*)
        Q = Q.view(b, n_q, h, self.d_k).transpose(1, 2) # (b, h, n_q, d_k)
        K = K.view(b, n_k, h, self.d_k).transpose(1, 2) # (b, h, n_k, d_k)
        V = V.view(b, n_v, h, self.d_v).transpose(1, 2) # (b, h, n_v, d_v)
        
        # 注意力分数 (Attention Scores): (b, h, n_q, n_k)
        # Fixed typo: 'math.sgrt' -> 'math.sqrt'
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k)
        
        mask = mask.unsqueeze(1)
        
        # mask中允许处为0，禁止处为 -inf (Allowed: 0, Forbidden: -inf)
        scores = scores + mask 
        
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        # (b, h, n_q, n_k)
        
        # 加权求和到值向量 (Weighted sum to value vector): (b, h, n_q, d_v)
        head_out = torch.matmul(attn, V)
        
        # 合并多头 -> (b, n_q, h*d_v)
        head_out = head_out.transpose(1, 2).contiguous().view(b, n_q, h * self.d_v)
        
        # 输出投影 -> (b, n_q, d_o)
        out = self.fc_o(head_out)
        
        return attn, out

# -------- 主代码 (Main Code) --------

batch = 10
num_head = 8
n_q, n_k, n_v = 4, 4, 4

# Fixed 'dimension g' to 'dimension_q'
dimension_q = dimension_k = 128
dimension_v = 64

# Fixed typo '16，16' (Chinese comma) to '16, 16'
d_k, d_v, d_o = 16, 16, 8 

# Fixed typo '9' to 'q'
q = torch.randn(batch, n_q, dimension_q)
k = torch.randn(batch, n_k, dimension_k)
v = torch.randn(batch, n_v, dimension_v)

# 构造一个上三角因果mask (对未来位置置 -inf)
# Construct an upper triangular causal mask (set future positions to -inf)
mask = torch.full((batch, n_q, n_k), float('-inf'))
mask = torch.triu(mask, diagonal=1) # 保留对角线以下为0，对角线上方为 -inf

mha = MHA(num_head, dimension_k, dimension_v, d_k, d_v, d_o)
attention, output = mha(q, k, v, mask)

print("Attention Size:", attention.size())
print("Output Size:", output.size())