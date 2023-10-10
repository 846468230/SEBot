import torch
import torch.nn as nn


# 多头自注意力机制实现
class SelfAttention(nn.Module):

    def __init__(self, dim, heads=8):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        assert (
            self.head_dim *
            heads == dim), "Embedding dimension needs to be divisible by heads"

        self.values = nn.Linear(self.dim,
                                self.heads * self.head_dim,
                                bias=False)
        self.keys = nn.Linear(self.dim, self.heads * self.head_dim, bias=False)
        self.queries = nn.Linear(self.dim,
                                 self.heads * self.head_dim,
                                 bias=False)
        nn.init.xavier_uniform_(self.values.weight)
        nn.init.xavier_uniform_(self.keys.weight)
        nn.init.xavier_uniform_(self.queries.weight)

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        # Split the embedding into self.heads different pieces
        # [batch_size, seq_len, dim] -> [batch_size, seq_len, heads, head_dim]
        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        # [batch_size, seq_len, heads, head_dim] -> [batch_size, heads, seq_len, head_dim]
        values = values.reshape(batch_size, seq_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_len, self.heads, self.head_dim)
        queries = queries.reshape(batch_size, seq_len, self.heads,
                                  self.head_dim)

        # Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm
        # [batch_size, heads, seq_len, head_dim] * [batch_size, heads, head_dim, seq_len]
        # -> [batch_size, heads, seq_len, seq_len]
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        # [batch_size, heads, seq_len, seq_len]
        attention = torch.softmax(energy / (self.dim**(1 / 2)), dim=3)

        # attention shape: [batch_size, heads, seq_len, seq_len], values shape: [batch_size, seq_len, heads, head_dim]
        # -> [batch_size, seq_len, heads, head_dim]
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            batch_size, seq_len, self.heads * self.head_dim)

        # [batch_size, seq_len, heads, head_dim] -> [batch_size, seq_len, dim]
        return out


if __name__ == '__main__':
    # 测试
    x = torch.randn(64, 10, 512)
    self_attention = SelfAttention(512)
    out = self_attention(x)
    print(out.shape)  # torch.Size([64, 10, 512])
