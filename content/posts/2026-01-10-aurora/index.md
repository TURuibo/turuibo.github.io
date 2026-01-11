---
title: "AURORA 1.3B: A foundation model for the Earth system"
date: 2026-01-10
draft: false
tags: ["foundation model", "multi-modal", "Microsoft Research", "AI for science"]
description: "A 1.3B encoder-decoder Transformer-based foundation model from Microsoft research AI for science."
ShowToc: true
TocOpen: true
ShowReadingTime: true
math: true
---
With a major impact of climate change, we are facing more and more challenging and extreme weather. During the New Year period of 2026, the named winter event Anna hit Sweden with strong winds and heavy snowfall. This leads to power outages, disrupted railway traffic, dangerous road conditions, etc. In Gävleborg in particular, it caused severe disruption—including widespread outages and major travel and transport paralysis—prompting authorities to issue a red warning. Can we know better where, when, and how such events will come for a better preparation for the future? We are expecting an answer from AI for science. And Microsoft research AI for science provides a foundation model AURORA for this.
With a major impact of climate change, we are facing more and more challenging and extreme weather. During the New Year period of 2026, the named winter event Anna hit Sweden with strong winds and heavy snowfall. This leads to power outages, disrupted railway traffic, dangerous road conditions, etc. In Gävleborg in particular, it caused severe disruption—including widespread outages and major travel and transport paralysis—prompting authorities to issue a red warning. Can we know better where, when, and how such events will come for a better preparation for the future? We are expecting an answer from AI for science. And Microsoft research AI for science provides a foundation model [AURORA](https://doi.org/10.1038/s41586-025-09005-y) for this.

In this post, let's learn about AURORA and discuss: what is a foundation model? What are we expecting from foundation models? What are beneficial discussions for the community? What is beneficial analysis in experiments? What are the other well known baseline models? Should we worry about data for AI for science compared with web-image-text for foundation models? A thought about finetuning regime, a thought about multi-task design. not so multi-tasks like text generation or mask text generation.

## Problem

Prediction accuracy and computational efficiency 

Computationally demanding: purpose-built supercomputers
Complexity: maintained, engineering
Accuracy: Approximation with sub-grids

Advantages of AI-based methods
Focus on global medium-range weather forecasting at 0.25 resolution
Leave the others like ocean dynamics, wave modelling, and atmospheric chemistry, and complex extreme weather prediction for future works

Not so convincing ...

## Model

### Deep Neural network architecture

Perceiver-based encoder-decoder:


Reason for why using Perceiver-based encoder:

* multi-modality and dimensionality

* not sure performance why, like diffusion models can not handle high-dimensionality 

A processor: Swin Transformer

* not sure why so, vs plain Transformer?

Training strategy: MAE for pretraining, one-step prediction

* this can be too small steps, what if high-order dynamics
* Like Latent diffusion for improvement?
* More probabilistic capture
* This is not multi-task, it is purely prediction...

Training recipes:

* Adam,
* step size, ...

Datasets:

* Multi-modality data with different pressures and resolutions
  


[![Diagram of Aurora, a flexible 3D foundation model of the atmosphere: pretrained on heterogeneous datasets (ERA5, CMIP6, GFS) and then fine-tuned with LoRA for operational forecasting at different resolutions; the model uses 3D Perceiver encoders/decoders around a 3D Swin Transformer U-Net to predict atmospheric fields from time T to T+1.](figures/image.png)](https://www.microsoft.com/en-us/research/blog/introducing-aurora-the-first-large-scale-foundation-model-of-the-atmosphere/)

*Figure: Aurora workflow and architecture (pretraining → fine-tuning & inference). Source: [Microsoft Research blog post](https://www.microsoft.com/en-us/research/blog/introducing-aurora-the-first-large-scale-foundation-model-of-the-atmosphere/).*


Some facts about training 
Some facts about performance

## References

1. Bodnár, C., et al. (2025). [A Foundation Model for the Earth System](https://doi.org/10.1038/s41586-025-09005-y). *Nature*.
