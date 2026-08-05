# Latent Personality Alignment (LPA)

Code for the paper **"Efficient Safety Alignment of Language Models via Latent Personality Traits"**, published at the **Conference on Language Modeling (COLM) 2026**.

**Mohamed Amine Merzouk**<sup>1,2</sup>, **Nolan Smyth**<sup>1,4</sup>, **Damiano Fornasiere**<sup>3</sup>, **Linh Le**<sup>5</sup>, **David Williams-King**<sup>5,6</sup>, **Adam Oberman**<sup>1,2,3</sup>

<sup>1</sup>Mila – Quebec AI Institute &nbsp;·&nbsp; <sup>2</sup>McGill University &nbsp;·&nbsp; <sup>3</sup>LawZero &nbsp;·&nbsp; <sup>4</sup>Université de Montréal &nbsp;·&nbsp; <sup>5</sup>Lida Safety &nbsp;·&nbsp; <sup>6</sup>ERA

📄 [arXiv](https://arxiv.org/abs/2607.07918) &nbsp;·&nbsp; [OpenReview](https://openreview.net/forum?id=2521)

---

## Overview

Latent Adversarial Training (LAT) is among the most effective defenses against jailbreak attacks, but it requires thousands of explicit harmful prompts paired with refusals, plus a supervised fine-tuning stage to recover the utility it costs.

**LPA replaces that harmful-prompt corpus with a compact set of harm-agnostic statements** drawn from the [IPIP](https://ipip.ori.org/) psychometric item pool. The model is adversarially trained, in latent space, to disagree with statements describing undesirable personality traits — statements that never mention harm, refusal, or safety. This is enough to drive HarmBench attack success rates to near zero across direct requests and five jailbreak methods while preserving utility, using ~75× fewer training examples and no supervised recovery stage. Training completes in minutes on a single GPU.

The method builds on the LAT implementation of [Sheshadri et al.](https://github.com/aengusl/latent-adversarial-training) and on the evaluation harness from [magikarp01/tasks](https://github.com/magikarp01/tasks), which we use through a [fork](https://github.com/mamerzouk/tasks).

## Repository layout

```
latent_at/
  lat_training.py      Entry point: builds the model, data, and trainer
  lat_methods.py       LAT trainer and the inner-loop PGD adversary
  lat_datasets.py      Chat-template formatting and batching
  lat_helpers.py       Loss computation (toward / away / SFT terms)
  laa/attacks.py       Latent-space attack hooks
  lat_config.json      Default training hyperparameters
eval.py                HarmBench ASR + utility evaluation of a checkpoint
data/                  Personality-statement datasets, one CSV per configuration
system_prompt/         System prompts used for training and evaluation
install_tasks_from_github.sh
```

## Installation

```bash
git clone https://github.com/mamerzouk/latent-personality-alignment.git
cd latent-personality-alignment

python -m venv env && source env/bin/activate
pip install -r requirements.txt
pip install fastchat

# Evaluation harness (HarmBench attacks + utility benchmarks)
bash install_tasks_from_github.sh
```

`install_tasks_from_github.sh` clones [mamerzouk/tasks](https://github.com/mamerzouk/tasks) into `tasks/` — our fork of [magikarp01/tasks](https://github.com/magikarp01/tasks), pinned to the version these experiments were run against.

A GPU with at least ~40 GB of memory is recommended for 7–8B models; all experiments in the paper ran on a single L40S.

## Training data

`data/` contains one CSV per training configuration, all drawn from the public-domain [IPIP item pool](https://ipip.ori.org/). Each row is a personality statement with the two competing completions:

```csv
prompt,rejected,chosen
I insult people.,I agree with this statement.,I do not agree with this statement.
I take advantage of others.,I agree with this statement.,I do not agree with this statement.
```

| Column | Role |
|---|---|
| `prompt` | The IPIP personality statement |
| `rejected` | The completion the PGD adversary is optimized **toward** |
| `chosen` | The completion the model is trained to produce **despite** the perturbation |

Nothing in any of these files mentions harm, refusal, or safety.

| File | Statements | Configuration |
|---|---:|---|
| `negative_only.csv` | 67 | **Main configuration.** Undesirable-trait statements, each paired with *"I do not agree with this statement."* |
| `positive_only.csv` | 58 | Desirable-trait statements, paired with *"I agree with this statement."* |
| `positive_and_negative.csv` | 125 | Both of the above combined |
| `inverted.csv` | 125 | Labels reversed: agree with undesirable traits, disagree with desirable ones |
| `shuffled.csv` | 125 | Statement–completion pairings randomly shuffled |
| `irrelevant.csv` | 121 | Traits unrelated to safety: artistic interests, intellect, sociability |
| `all_ipip.csv` | 3766 | The full IPIP item pool |

The first three isolate the effect of statement valence; the last four are the trait-selection ablations. `negative_only.csv` is what reproduces the headline results — training on the others reaches comparable ASR but at a substantial cost in utility.

A benign dataset is also loaded, for the optional SFT term. The paper uses [`LLM-LAT/benign-dataset`](https://huggingface.co/datasets/LLM-LAT/benign-dataset), with its loss coefficient set to `0` — so it does not affect the default LPA objective.

## Training

```bash
python -m latent_at.lat_training \
    --model_name Qwen/Qwen3-8B \
    --harmful_dataset data/negative_only.csv \
    --benign_dataset LLM-LAT/benign-dataset \
    --system_prompt_path system_prompt/alpha.txt \
    --lat_config_path latent_at/lat_config.json \
    --cache_dir cache \
    --project_name lpa-qwen3-8b \
    --batch_size 16 \
    --num_steps 30 \
    --N_checkpoints 10 \
    --epsilon 6 \
    --def_sft 0
```

Checkpoints are written to `cache/<project_name>_<timestamp>/checkpoint_<N>/` as LoRA adapters, alongside a `parameters.json` recording the run configuration. Pass `--timestamp` to set the run ID explicitly; otherwise one is generated.

Training is logged to Weights & Biases. Use `--wandb-offline`, or `export WANDB_MODE=disabled`, to turn that off.

### Supported model families

The chat template is selected from the model name: `Llama-2`, `Llama-3`, `Qwen`, `Mistral`, `Olmo`/`OLMo`, and `zephyr`. Anything else falls back to a generic template. The paper reports Qwen3-8B (main results), Llama-3-8B, and Mistral-7B-Instruct-v0.3.

### Hyperparameters

Defaults live in `latent_at/lat_config.json` and can each be overridden on the command line:

| Parameter | Value | Flag |
|---|---|---|
| LoRA rank | 64 | *(set in code)* |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj` | *(set in code)* |
| PGD injection points | `["embedding", 8, 16, 24, 30]` | `--pgd_layers` |
| PGD constraint (ℓ₂) | ε = 6 | `--epsilon` |
| PGD iterations per step | 16 | `--pgd_iterations_per_step` |
| Model updates per step | 4 | `--model_iterations_per_step` |
| Training steps | 30 (Qwen3-8B, Llama-3-8B); 8 (Mistral) | `--num_steps` |
| Adversary loss coefficients | toward 0.5 / away 0.5 | `--adv_toward`, `--adv_away` |
| Defender loss coefficients | toward 0.5 / away 0.5 / sft 0 | `--def_toward`, `--def_away`, `--def_sft` |

One **training step** is 16 PGD iterations followed by 4 model update steps on a single batch. The base model is frozen throughout; only the LoRA adapter is updated.

`--pgd_layers` takes a JSON string:

```bash
--pgd_layers '["embedding", 8, 16, 24, 30]'
```

## Evaluation

```bash
python -m eval \
    --project_name lpa-qwen3-8b \
    --run_id <timestamp> \
    --epoch 30
```

This loads `cache/<project_name>_<run_id>/checkpoint_<epoch>/`, applies the LoRA adapter to the base model named in `parameters.json`, and runs:

- **HarmBench** — direct requests plus GCG, PAIR, AutoPrompt, AutoDAN, and TAP, scored with the HarmBench classifier (`cais/HarmBench-Llama-2-13b-cls`). GCG and AutoPrompt run white-box against the defended model.
- **Utility** — MMLU, HellaSwag, Winogrande, SciQ, and Lambada.

Omit `--epoch` to evaluate the final adapter rather than an intermediate checkpoint. Results are logged to W&B and cached under the checkpoint directory.

Evaluation is run with the minimal system prompt (`system_prompt/minimal.txt`) so that the safety behavior measured comes from training rather than from prompting.

## System prompts

| File | Used for |
|---|---|
| `alpha.txt` | Training prompt, framing the task as a personality self-assessment |
| `gamma.txt` | `alpha.txt` with the refusal-permission sentence removed — the ablation showing the safety effect does not depend on that clause |
| `beta.txt` | Minimal training instruction, for the system-prompt ablation |
| `minimal.txt` | `"You are a helpful assistant."` — used for all evaluations |
| `safety.txt` | Llama-2-style safety prompt, for baseline comparisons |
| `persist.txt` | Likert-scale personality assessment, for measuring trait stability |

## Citation

```bibtex
@inproceedings{merzouk2026lpa,
  title     = {Efficient Safety Alignment of Language Models via Latent Personality Traits},
  author    = {Merzouk, Mohamed Amine and Smyth, Nolan and Fornasiere, Damiano
               and Le, Linh and Williams-King, David and Oberman, Adam},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026},
  eprint    = {2607.07918},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2607.07918}
}
```

## Acknowledgments

This work was supported by the Natural Sciences and Engineering Research Council of Canada (NSERC), by Coefficient Giving, and by a Canada CIFAR AI Chair. This research was enabled in part by support provided by [Calcul Québec](https://www.calculquebec.ca) and the [Digital Research Alliance of Canada](https://alliancecan.ca).

## License

See [LICENSE](LICENSE).
