---
title: "Multi-Modal Vision Language Models"
date: 2026-01-26
draft: true
tags: ["foundation model", "multi-modality","vision language model","VQA","vision language reasoning", "video QA"]
description: "Read vision language models for handling multi-modality."
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---
## Overview

Vision-Language Models
The models are focused in this post are the ones

* image-text retrieval;
* visual question answering;
* visual reasoning;
* visual entailment;
* weakly-supervised grounding.

Furthermore, this post focused on vision-language models with cross-attention for multi-modal modelling. The models are introduced following this order.
1. ALBEF
2. (CoCa)
3. BLIP
4. Flamingo
5. (CLIP-ViT)

**ALBEF.**
This paper summarizes the related work in two categories regarding multi-modal modelling.

1. joint vision-language encoders with cross-attention for complex reasoning;
2. separate uni-modal encoders, like CLIP, for simple tasks like image-text retrieval.

ALBEF proposes to combine both categories of models with Image-Text Contrastive learning (ITC), Masked Language Modelling (MLM) and Image-Text Matching (ITM) objectives. Moreover, it proposes to use two important approximations with queuing memory bank [citation memory bank] and soft labels. As a result, it can use 2xA100 GPUs for 30 epochs of batch size 512.

As shown in Fig 1, the objectives are applied to both individual encoders before using cross-attention and to the final representation after using cross-attention.
It worths to note that contrastive learning requires large number of negative samples, like a huge batch in CLIP. Using memory bank saves computes by re-using the outputs from previous calls of teacher models as the approximation of the outputs of the current student model. Moreover, to handle the noisy labelling of web data, a mix of soft labels from teacher models and labels by annotators are used for training.
![Fig 1](figures/albef.png)
 *source ([citation ALBEF])*


## References
* Yu, J., Wang, Z., Vasudevan, V., Yeung, L., Seyedhosseini, M., & Wu, Y. (2022). [CoCa: Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917). *arXiv*. 
* Li, J., et al. (2021). [ALBEF: Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651). *NeurIPS*. 
* Alayrac, J.-B., et al. (2022). [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198). *NeurIPS*.
* Chen, J., et al. (2022). [VisualGPT: Data-Efficient Adaptation of Pretrained Language Models for Image Captioning](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_VisualGPT_Data-Efficient_Adaptation_of_Pretrained_Language_Models_for_Image_Captioning_CVPR_2022_paper.html). *CVPR*.
* Li, J., et al. (2022). [BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086). *arXiv*.
* Li, J., et al. (2023). [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597). *arXiv*.
* Xue, L., et al. (2024). [xGen-MM (BLIP-3): A Family of Open Large Multimodal Models](https://arxiv.org/abs/2408.08872). *arXiv*.
* Yu, J., et al. (2022). [CoCa: Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917). *arXiv*.
* Bai, J., et al. (2023). [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966). *arXiv*.
* Liu, H., et al. (2023). [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485). *arXiv*.
* Kim, W., et al. (2021). [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334). *ICML*.
