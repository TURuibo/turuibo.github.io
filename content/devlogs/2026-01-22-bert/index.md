---
title: "MHSA of BERT"
date: 2026-01-22
draft: false
tags: ["implementation", "multi-head self-attention", "bert"]
description: "An implementation of multi-head self-attention of BERT."
ShowToc: false
TocOpen: false
ShowReadingTime: false
math: true
---

## Multi-Head Self-Attention

The Transformer blocks remove casual masks while take the attention masks. The attention mask indicating the padding tokens has the shape (B,T) for batch size B and sequence length T. Such masks are applied before softmax and often mask over key instead of query in attention computation. Moreover, an interesting implementation for multi-head attention is to initialize one matrix for all head and then reshape-reorder for attention score computation.

```python
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention (bidirectional) with padding mask.

    Inputs:
        x: [B, T, H]
        attention_mask: [B, T] with 1 for real tokens and 0 for padding.

    Output:
        y: [B, T, H]

    Notes:
    - No causal masking (BERT is bidirectional).
    - Implement `scores.masked_fill(mask == 0, -1e4)` **before** softmax.
    - Apply dropout to attention probabilities and to the final output projection.
    """

    def __init__(self, hidden_size: int, num_heads: int, dropout_prob: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_drop = nn.Dropout(dropout_prob)
        self.proj_drop = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        Q, K, V = torch.split(self.qkv(x), self.hidden_size, dim=-1)
        B,T,D = x.shape
        Q= Q.reshape(B,T,self.num_heads,self.head_dim).permute(0,2,1,3)
        K= K.reshape(B,T,self.num_heads,self.head_dim).permute(0,2,1,3)
        V= V.reshape(B,T,self.num_heads,self.head_dim).permute(0,2,1,3)
        scores = Q @ K.permute(0,1,3,2) / (self.head_dim)**0.5
        scores = scores.masked_fill(attention_mask.reshape(B,1,1,T) == 0, -1e4)
        probs = torch.softmax(scores, dim=-1)
        probs = self.attn_drop(probs)
        O = probs @ V
        O_ = self.out(O.permute(0,2,1,3).reshape(B,T,D))
        return self.proj_drop(O_)
```
