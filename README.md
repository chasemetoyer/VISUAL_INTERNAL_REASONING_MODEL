# Hypothesis

A transformer-based language model that is trained to generate and condition on an explicit internal visual latent representation, implemented as discrete image tokens within a unified autoregressive vocabulary, will achieve higher accuracy and robustness on spatial and visually grounded reasoning tasks than an architecture-matched text-only baseline.

Furthermore, the model’s performance on these tasks will causally depend on the internal visual representation, such that corruption or removal of the generated visual tokens at inference time will result in a statistically significant degradation in task performance.

# Abstract

Large language models (LLMs) exhibit strong performance on linguistic reasoning tasks but continue to struggle with problems requiring spatial and visual reasoning, often relying on shallow textual heuristics rather than grounded internal representations. In this work, we investigate whether equipping a transformer-based language model with an explicit internal visual latent representation improves performance on visually grounded reasoning tasks.

We propose a unified decoder-only architecture in which discrete image tokens, obtained from a pretrained vector-quantized autoencoder, are incorporated directly into the model’s vocabulary alongside standard text tokens. The model is trained autoregressively to generate an intermediate sequence of image tokens—interpreted as an internal “imagined” visual state—prior to producing a textual answer. This design enables the model to construct and condition on a visual representation without invoking external image generation or perception modules at inference time.

To enable controlled evaluation, we introduce a synthetic data generation pipeline that produces triplets of (text prompt, visual intermediate, text answer) for spatial manipulation and object-relation tasks. We compare this “imaginer” model against a text-only baseline with matched architecture and parameter count. Causal ablations, including corruption and replacement of intermediate visual tokens at inference time, are used to assess whether performance depends on the generated visual representation.

Our results demonstrate that models trained with forced internal visual intermediates outperform text-only baselines on spatial reasoning tasks and exhibit significant performance degradation when the imagined visual tokens are disrupted, indicating that the visual representation is causally utilized. These findings suggest that explicit internal visual latents can serve as a useful inductive bias for improving reasoning in transformer-based language models, and motivate further exploration of structured intermediate representations for multimodal and world-model learning.
