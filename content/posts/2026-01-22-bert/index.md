---
title: "BERT"
date: 2026-01-22
draft: false
tags: ["foundation model", "language model"]
description: "An effective Transformer-based method for pre-training and finetuning regime."
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---
BERT takes the encoder of Transformer ([Vaswani et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)) pre-trained with masked language model task and next sentence prediction. It can be finetuned on downstream tasks and achieves state-of-the-art performance. MLM plays an important role in self-supervised learning, and inspired MAE-ViT ([He et al., 2022](https://arxiv.org/abs/2111.06377)). Another important self-supervised learning task is contrastive learning, e.g., used by DINO ([Caron et al., 2021](https://arxiv.org/abs/2104.14294)). While the representation learned with MLM in general requires finetuning for downstream tasks, contrastive learning leads to better zero-shot, few-shot or in-context learning performance. But the simplicity and efficiency of MLM makes it as a compelling method for pretraining.

## Problem

The starting point of BERT is the limitation of left-to-right Transformers, such as GPT ([Radford et al., 2018]). Because first of all, limiting deep learning models is not a good practice, e.g., the invertibility of deep neural network in normalizing flows. Secondly, the left-to-right inductive bias doesn't fit all NLP tasks. Therefore, BERT removes the constraints and allow attention layers to compute attentions across all tokens within a sequence of tokens.

## Model

BERT consists of embedding layers, Transformer blocks, and pre-training task heads, including MLM and NSP.

### Embedding layers

The embedding layer handles three inputs

* Ids of tokens, e.g., 1, 7, 8, ...;
* Positions of tokens in a sequence with maximum length, e.g., 1,..., T;
* Token types, either from the first sentence or the second sentence, e.g., 0,0,0,....,1,1.

For each input, the embedding layer produces a vector as the embedding of which the hidden size is the same for all inputs,  and then sums up all the embedding as the input to the Transformer blocks.

``` python
self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
```

### Transformer blocks

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

## Pre-training and Fine-tuning
MLM and NSP are used for pre-training; however, NSP is shown to be less effective when scaling up by adding more data in a batch and more data ([Liu et al., 2019](https://arxiv.org/abs/1907.11692)). From this observation, it is interesting to see the behavior change when scaling up. Simple methods can work better.

**MLM.** Weight tying is used and shown to be effective in language modelling, which uses the transpose of the embedding layer of input tokens as the output layer. It saves the number of parameters and enforce the hypothesis that the output embedding space should be similar as the input embedding space.

**Finetuning.** 
Downstream task heads are added on top of the last layer and all parameters are finetuned. LoRA was not developed at that time and the parameter number is still feasible to be finetuned, like less than 1 billion.

![BERT Pre-training and Fine-tuning](figures/BERT_masked_language_modelling_task.png)
*Figure 1: Masked language modelling tasks. Source: [Wikipedia](https://en.wikipedia.org/wiki/BERT_(language_model)).*

![BERT Input Representation](figures/BERT_next_sequence_prediction_task.png)
*Figure 2: Next sentence prediction task. Source: [Wikipedia](https://en.wikipedia.org/wiki/BERT_(language_model)).*

## Discussion

Masked value prediction and removing the constraints on attention layers make the model assumptions hold for other domains, like image, tabular data, time-series data, as well. So regarding general self-supervised learning with finetuning for downstream tasks, it will be more reasonable to start from BERT structure than GPT structure.

## Further Reading

* DINO, the other way of self-supervised learning for representation learning;
* MAE-ViT, the application of MLM in images;
* RoBERTa, scale up BERT with better practices.


## References

* He, K., et al. (2022). [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377). *CVPR*.
* Caron, M., et al. (2021). [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294). *ICCV*.
* Liu, Y., et al. (2019). [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). *arXiv*.
* Radford, A., et al. (2018). [Improving Language Understanding by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf). *OpenAI*.
* Vaswani, A., et al. (2017). [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html). *NeurIPS*.
