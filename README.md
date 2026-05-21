# Label-Free Cross-Task LoRA Merging with Null-Space Compression [CVPR 2026]

**Official PyTorch implementation of [*Label-Free Cross-Task LoRA Merging with Null-Space Compression*] [CVPR 2026].**

Wonyoung Lee\* & Wooseong Jeong\* & Kuk-Jin Yoon, Korea Advanced Institute of Science and Technology (KAIST)

Model merging combines independently fine-tuned checkpoints without joint multi-task training. In the era of foundation-model, fine-tuning with Low-Rank Adaptation (LoRA) is prevalent, making LoRA merging a promising target. Existing approaches can work in homogeneous settings where all target tasks are classification but often fail when tasks span classification and regression. Approaches using entropy-based surrogates do not apply to regression and are costly for large language models due to long token sequences. We introduce Null-Space Compression (NSC) Merging, a label-free, output-agnostic method that sets merge weights from adapter geometry. Our key observation is that during LoRA finetuning the down-projection factor $A$ in $\Delta W = BA$ compresses its null space, and the compression correlates with performance. NSC uses this as an optimization signal for merging that can generalize across classification, regression, and sequence generation. NSC achieves state-of-the-art performance across twenty heterogeneous vision tasks with balanced gains where prior methods overfit subsets of tasks. It also outperforms baselines on six NLI benchmarks and on vision-language evaluations for VQA and image captioning, demonstrating scalability and effectiveness.

---

## Getting Started

### 1. Environment Setup

The environment is provided as a conda spec in `environment.yml`. It pins CUDA 12.1, Python 3.11, PyTorch 2.5.1, `transformers`, `peft`, and the vision-language evaluation dependencies (`pycocoevalcap`, `pycocotools`, etc.).

```bash
conda env create -f environment.yml
conda activate nsc-merging
```

### 2. Download LoRA Adapters

Pre-trained LoRA adapters for the six LLaVA-1.5-7B tasks (VizWiz VQA, IconQA, ChartQA, DocVQA, COCO Captioning, Flickr30k Captioning) are hosted on the Hugging Face Hub. The download script places them under `./checkpoints/llava/`.

```bash
bash scripts/download.sh
```

This requires the `hf` CLI (bundled with `huggingface_hub`). If you have not logged in yet, run `hf auth login` first.

After running, the directory layout should be:

```
checkpoints/llava/
├── vizwiz_vqa/
├── iconqa/
├── chartqa/
├── docvqa/
├── coco/
└── flickr/
```

### 3. Run the VLM Merging Experiment

The main entry point is `src/merge.py`. The provided launcher merges all six LLaVA adapters with NSC and evaluates the merged model:

```bash
bash scripts/run_vlm.sh
```

Which expands to:

```bash
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

python src/merge.py \
  --run_name "six_llava_rank16" \
  --model_dirs llava_cfg/vizwiz_vqa_rank16.yaml \
               llava_cfg/iconqa_rank16.yaml \
               llava_cfg/chartqa_rank16.yaml \
               llava_cfg/docvqa_rank16.yaml \
               llava_cfg/coco_rank16.yaml \
               llava_cfg/flickr_rank16.yaml \
  --batch_size 1 --lamda 2.4 --kernel --learning_rate 3e-4
```

#### Key arguments

| Argument           | Description                                                                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `--run_name`       | Identifier used for logging and output directories.                                                                                            |
| `--model_dirs`     | One YAML config per task to be merged (see `llava_cfg/`). The set of files determines which adapters get merged.                               |
| `--lamda`          | Scaling factor applied to the merged update. `2.4` reproduces the six-task LLaVA setting.                                                      |
| `--learning_rate`  | Learning rate for the merging-coefficient optimization.                                                                                        |
| `--batch_size`     | Batch size used during merging / evaluation. For LLaVA-1.5-7B keep this at `1` to fit on a single GPU.                                         |
| `--kernel`         | Enables the NSC kernel merging objective (the proposed method). Omit it (and pass `--simple_merging`) to fall back to baseline simple merging. |
| `--simple_merging` | Run baseline simple merging with fixed coefficients instead of optimization.                                                                   |
| `--num_steps`      | Number of optimization steps (default `500`).                                                                                                  |

#### Per-task YAML config

Each file under `llava_cfg/` describes a single adapter and how to evaluate it. Example (`llava_cfg/vizwiz_vqa_rank16.yaml`):

```yaml
run_name: ${model.name}_${data.name}_${task.name}
processor:
  size: 336
  use_fast: true
model:
  id: llava-hf/llava-1.5-7b-hf
  name: llava_1_5
  flash_attention: false
  bf16: true
task:
  name: vqa
data:
  name: VizWiz_VQA
dataloader:
  train_batch_size: 1
  train_num_workers: 0
  val_batch_size: 2
  val_num_workers: 4
adapter_save_path: ./checkpoints/llava/vizwiz_vqa/
device: auto
```

To merge a different subset of tasks, pass the corresponding YAML files to `--model_dirs`. To evaluate on your own LoRA adapter, create a new YAML following the schema above and point `adapter_save_path` at the adapter directory.

### 4. Hardware

Experiments were run on a single NVIDIA GPU with at least 24 GB of memory (for `batch_size=1`, `bf16=true`). Set `CUDA_VISIBLE_DEVICES` in `scripts/run_vlm.sh` to choose a different device.

---

## Repository Layout

```
.
├── environment.yml         # conda spec for reproducible setup
├── llava_cfg/              # per-task YAML configs for the six LLaVA tasks
├── scripts/
│   ├── download.sh         # downloads the released LoRA adapters
│   └── run_vlm.sh          # main launcher for the VLM merging experiment
└── src/
    ├── merge.py            # entry point: see `python src/merge.py --help`
    ├── merge_strategy.py   # NSC and baseline merging strategies
    ├── baselines.py        # baseline merging methods (TA, TIES, ...)
    ├── models/, vlm/, data/, utils.py, ...
```

For full argument documentation, run:

```bash
python src/merge.py --help
```

---

## Contact

- Wonyoung Lee: wylee@kaist.ac.kr
- Wooseong Jeong: stk14570@kaist.ac.kr

## Reference

If you find this code useful, please cite the following paper:

```bibtex
@article{lee2026label,
  title={Label-Free Cross-Task LoRA Merging with Null-Space Compression},
  author={Lee, Wonyoung and Jeong, Wooseong and Yoon, Kuk-Jin},
  journal={arXiv preprint arXiv:2603.26317},
  year={2026}
}
```
