# GFT: Graph-knowledge Fine-Tuning for Explainable DR Diagnosis (MICCAI 2025)

> 🔎 **New:** our latest work on **trust-in-AI for DR diagnosis from OCTA** (survey).  
> **[Link coming soon — add URL here]**

This repo releases a clean, working reference for our MICCAI paper, including a minimal GNN pipeline, integrated-gradients attribution, instruction data synthesis, and two-stage VLM fine-tuning & demo. The code favors clarity and reproducibility over flash.

## Figures (placeholders)
- **Figure 1. Method overview** — _Insert figure from paper here_
- **Figure 2. Instruction data examples** — _Insert figure from paper here_
- **Figure 3. Interpretability comparison** — _Insert figure from paper here_

## Highlights
- Biology-informed graph → GNN staging → IG attribution → vision-language tuning
- Two-stage finetuning script for Llama 3.2 Vision (Unsloth)
- Minimal demo app with Gradio

## Quickstart
```bash
# 1) env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) run a tiny GNN sanity-check on toy data
python -m gft.training.train_gnn --data-dir examples/sample_data --epochs 2

# 3) export IG table + synthetic Q&A (no external APIs)
python scripts/generate_instructions.py --data-dir examples/sample_data --out instructions.jsonl

# 4) stage 1 finetune (classification + short rationale)
python -m gft.training.ft_stage1 --model unsloth/Llama-3.2-11B-Vision-Instruct --data instructions.jsonl --out checkpoints/stage1

# 5) stage 2 finetune (region-aware Q&A)
python -m gft.training.ft_stage2 --model checkpoints/stage1 --data instructions.jsonl --out checkpoints/stage2

# 6) launch demo
python -m gft.inference.deploy --model checkpoints/stage2
```

## Repo layout
```
src/gft
  data/                 # image reader, toy labels
  graphs/               # graph builder + IG attribution
  models/               # light GraphSAGE
  training/             # GNN trainer + VLM finetune stages
  inference/            # demo + single-image inference
  utils/                # misc helpers
scripts/
  generate_instructions.py
examples/
  sample_data/          # placeholder
  configs/default.yaml
```

## Notes
- The **graph builder** here is a minimal stand-in: it splits an OCTA into tiles and extracts basic patch stats, so the whole stack runs end-to-end. Replace it with your vessel/ICA/FAZ graph when ready.
- Finetuning uses **Unsloth** + **TRL**. If you prefer another stack, only the wrappers need to change.
- The demo streams tokens and accepts image + prompt.

## License
MIT
