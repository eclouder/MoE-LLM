import torch
from torch import nn
import math
from einops import repeat


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Attention(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, capacity: float):
        super(Attention, self).__init__()
        self.dim = dim
        self.capacity = capacity
        self.router = nn.Linear(dim, 1)
        self.inner_dim = hidden_dim

        self.scale = 1 / torch.sqrt(torch.tensor(self.inner_dim, dtype=torch.float32))
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

    def forward(self, x, select_x_mask, router_k, topk_index):
        b, l, d = x.size()

        select_x = torch.masked_select(x, select_x_mask).view(b, router_k, d)
        output = torch.zeros(b, l, d)
        q_weight = self.to_q.weight.view(1, self.inner_dim, self.dim).repeat(b, 1, 1)
        k_weight = self.to_k.weight.view(1, self.inner_dim, self.dim).repeat(b, 1, 1)
        v_weight = self.to_v.weight.view(1, self.inner_dim, self.dim).repeat(b, 1, 1)
        s_q_weight = torch.masked_select(q_weight, select_x_mask).view(b, router_k, d)
        s_k_weight = torch.masked_select(k_weight, select_x_mask).view(b, router_k, d)
        s_v_weight = torch.masked_select(v_weight, select_x_mask).view(b, router_k, d)
        q = select_x * s_q_weight
        k = select_x * s_k_weight
        v = select_x * s_v_weight
        score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_weight = torch.softmax(score, dim=-1)
        s_output = torch.matmul(attention_weight, v)
        for i in range(b):
            output[i, topk_index[i], :] = s_output[i]
            output[i, topk_index[i], :] = x[i, ~topk_index[i], :]

        return output


class Block(nn.Module):

    def __init__(self, dim, inner_dim, capacity, resid_pdrop):
        super().__init__()
        self.router = nn.Linear(dim, 1)

        self.ln_1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, inner_dim, capacity)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.ModuleDict(dict(
            c_fc=nn.Linear(dim, 4 * dim),
            c_proj=nn.Linear(4 * dim, dim),
            act=NewGELU(),
            dropout=nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        b, l, d = x.size()

        router_weight = self.router(x).squeeze(2)
        router_k = math.ceil(l * self.capacity)
        topk_value, topk_index = torch.topk(router_weight, k=router_k, dim=1, sorted=True)
        topk_value_last = topk_value[:, -1].view(b, 1).repeat(1, l)  # b,l
        _select_x_mask = router_weight >= topk_value_last  # b,l
        select_x_mask = repeat(_select_x_mask, 'b l -> b l d', d=d)

        x = x + self.attn(self.ln_1(x), select_x_mask, router_k, topk_index)
        x = x + self.mlpf(self.ln_2(x))
        return x
