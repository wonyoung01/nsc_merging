# Label-Free Cross-Task LoRA Merging with Null-Space Compression [CVPR 2026]

**Official PyTorch implementation of [*Label-Free Cross-Task LoRA Merging with Null-Space Compression*] [CVPR 2026].**

Wonyoung Lee* & Wooseong Jeong* & Kuk-Jin Yoon, Korea Advanced Institute of Science and Technology (KAIST)

Model merging combines independently fine-tuned checkpoints without joint multi-task training. In the era of foundation-model, fine-tuning with Low-Rank Adaptation (LoRA) is prevalent, making LoRA merging a promising target. Existing approaches can work in homogeneous settings where all target tasks are classification but often fail when tasks span classification and regression. Approaches using entropy-based surrogates do not apply to regression and are costly for large language models due to long token sequences. We introduce Null-Space Compression (NSC) Merging, a label-free, output-agnostic method that sets merge weights from adapter geometry. Our key observation is that during LoRA finetuning the down-projection factor $A$ in $\Delta W = BA$ compresses its null space, and the compression correlates with performance. NSC uses this as an optimization signal for merging that can generalize across classification, regression, and sequence generation. NSC achieves state-of-the-art performance across twenty heterogeneous vision tasks with balanced gains where prior methods overfit subsets of tasks. It also outperforms baselines on six NLI benchmarks and on vision-language evaluations for VQA and image captioning, demonstrating scalability and effectiveness.

## We are preparing the code for public release. It will be available here soon.

## Contact
Wonyoung Lee: wylee@kaist.ac.kr
Wooseong Jeong: stk14570@kaist.ac.kr
