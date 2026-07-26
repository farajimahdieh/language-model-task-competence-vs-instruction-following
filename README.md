# Task Competence Is Not Instruction Following (https://arxiv.org/abs/2607.19608)

**Evaluating Instruction-Conflicting Behavior in Small Language Models**

This repository contains the code and evaluation pipeline for the paper *"Task Competence Is Not Instruction Following: Evaluating Instruction-Conflicting Behavior in Small Language Models"* by Mahdiyeh Farajidizaji and Vatsal Raina.

## Overview

Instruction tuning is meant to make language models follow user requests, yet it is unclear whether small models comply when an instruction **conflicts** with their usual task behavior. We study this by pairing each standard instruction with a conflicting **non-standard** one and measuring whether models follow the request or fall back on their learned task prior.

The key finding: **task competence and instruction following are distinct abilities.** Small models often stay competent (high standard accuracy) while routinely ignoring non-standard instructions, whereas larger models show a clear gap between the two settings. Reporting only standard accuracy hides these instruction-following failures.

## Tasks and Instructions

We evaluate the same phenomenon across three tasks. For each, a **standard** instruction matches the usual objective and a **non-standard** instruction asks the model to deviate in a well-defined way. All predictions are scored against the *original ground truth*, so a model that ignores the non-standard instruction still appears "accurate".

| Task | Standard instruction | Non-standard instruction |
| --- | --- | --- |
| Multiple-choice QA (MCQA) | Select the correct option | Select an incorrect option |
| Sentiment classification | Report the true sentiment | Report the opposite sentiment |
| Mathematical QA | Return the correct answer | Return twice the answer |

## Metrics

- **Standard accuracy** — fraction answered correctly under the standard instruction (measures task competence; higher is better).
- **Non-standard accuracy** — fraction that still produce the ground-truth answer under the non-standard instruction (higher is *worse*, as it means the instruction was ignored).
- **Instruction-Following Failure Rate (IFFR)** — how often a model defaults to the standard answer despite being told to deviate, computed *only* over examples answered correctly under the standard instruction:

  ```
  IFFR = P(ŷ_ns = y | ŷ_s = y)
  ```

  where `y` is the ground truth and `ŷ_s`, `ŷ_ns` are the predictions under the standard and non-standard instructions. Lower IFFR indicates better compliance. Because IFFR conditions on items the model can already solve, it isolates instruction following from raw task competence.

## Models

We evaluate instruction-tuned **Qwen3.5** checkpoints across a same-family scale range, so behavioral differences can be attributed to scale rather than divergent pretraining or alignment:

- `Qwen/Qwen3.5-0.8B`
- `Qwen/Qwen3.5-2B`
- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-9B`
- `Qwen/Qwen3.5-27B`

All experiments use **deterministic (greedy) decoding** (`temperature=0` / `do_sample=False`) with the Qwen chat template and `enable_thinking=False`.

## Datasets

Each task uses three datasets:

| Task | Dataset | # Examples |
| --- | --- | --- |
| MCQA | RACE-Middle | 1,436 |
| MCQA | ARC-Challenge | 1,165 |
| MCQA | OpenBookQA-Main | 500 |
| Sentiment | Multi-class Sentiment | 3,276 |
| Sentiment | Rotten Tomatoes | 1,066 |
| Sentiment | FinancialPhraseBank | 197 |
| Math QA | MAWPS | 355 |
| Math QA | Calc-asdiv-a | 1,218 |
| Math QA | MultiArith | 180 |

For sentiment classification, neutral examples are removed so the "opposite" sentiment is well defined (binary positive/negative). For ARC-Challenge, only examples with exactly four options are kept.

## Evaluation Pipeline

All three tasks share a decoder-only Qwen backbone and differ only in how the next-token distribution is reduced to a prediction:

- **MCQA** — read the first-position logits and restrict them to the option tokens `{A, B, C, D}`; the prediction is the restricted argmax. This guarantees a valid, in-set answer and puts the standard and non-standard settings on an identical footing.
- **Sentiment classification** — the same closed-set decoding, restricted to the label tokens `{positive, negative}`.
- **Mathematical QA** — greedily decode a full response, then extract and normalize the final numeric value and test it for exact equality against the gold answer.

Each prompt follows a shared template — an instruction line, the task input, a short output-format constraint, and a final `Answer:` cue — and every model is evaluated with **three prompt variants** per task (only the instruction line changes). Reported numbers are the mean over the three variants.

## Key Results

- **MCQA & Math QA:** standard accuracy rises with scale, while non-standard accuracy and IFFR drop sharply — the gap between the two settings widens with model size.
- **Sentiment:** standard accuracy is high and flat across sizes, but smaller models frequently report the true sentiment when asked for the opposite; larger models comply far more reliably.
- **IFFR falls steeply with scale** across all three tasks (e.g., MCQA RACE-M drops from ~95.7 at 0.8B to ~4.2 at 27B). The main exception is MultiArith, where IFFR rises again at 27B, showing scale reduces but does not eliminate reversion to the learned task prior.
- IFFR is most sensitive to instruction wording in the 4B–9B range, where models transition from ignoring to following the non-standard instruction.

See the paper (Tables 3–8, Figures 3–4) for full per-dataset and per-variant results.

## Repository Structure

```
.
├── mcqa/            # MCQA evaluation (restricted-argmax over A/B/C/D)
├── sentiment/       # Sentiment classification (restricted-argmax over labels)
├── math_qa/         # Mathematical QA (numeric extraction + matching)
├── prompts/         # Standard and non-standard instruction variants
└── README.md
```

> Adjust the structure above to match the actual layout of the code.

## Setup

```bash
git clone https://github.com/farajimahdieh/language-model-task-competence-vs-instruction-following.git
cd language-model-task-competence-vs-instruction-following
pip install -r requirements.txt
```

Local inference uses Hugging Face Transformers; the 9B and 27B models (and all MCQA runs) were evaluated through the [DeepInfra](https://deepinfra.com/) OpenAI-compatible chat-completions API. Set the relevant API key before running API-backed evaluations:

```bash
export DEEPINFRA_API_TOKEN=your_token_here
```

## Citation

If you use this code or the findings in your work, please cite:

```bibtex
@misc{farajidizaji2026taskcompetenceinstructionfollowing,
      title={Task Competence Is Not Instruction Following: Evaluating Instruction-Conflicting Behavior in Small Language Models}, 
      author={Mahdiyeh Farajidizaji and Vatsal Raina},
      year={2026},
      eprint={2607.19608},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.19608}, 
}
```

## Contact

- Mahdiyeh Farajidizaji — mahdiehfaraji194@gmail.com
- Vatsal Raina — vatsal@aptaai.com
