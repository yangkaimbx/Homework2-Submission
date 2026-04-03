# Week 2: LLM Architecture & Training Lifecycle

**Course:** Machine Learning Engineer in the Generative AI Era  
**Week:** 2 of 10  
**Topic:** Transformers, Pretraining Pipeline, Fine-Tuning, Alignment, Test-Time Scaling

---

## Overview

This homework takes you through the full lifecycle of a large language model — from the Transformer architecture that powers it, through the data pipeline that feeds it, to the alignment techniques that make it useful.

By the end you will have:
- Implemented scaled dot-product attention from scratch and visualised multi-head attention
- Compared three major tokenisation systems and understood their cost implications
- Built a complete data collection pipeline (web scraping, PDF OCR, ASR)
- Cleaned, deduplicated, and anonymised a text corpus
- Explored fine-tuning and alignment concepts (SFT, LoRA, DPO, PPO)
- Used chain-of-thought and extended thinking to improve LLM output quality

---

## Learning Objectives

1. Understand the Transformer architecture (attention, FFN, positional encoding, LayerNorm)
2. Compare tokenisation algorithms (BPE, tiktoken, SentencePiece) and their cost impact
3. Build a pretraining data pipeline: web scraping → PDF OCR → ASR → cleaning
4. Apply MinHash deduplication and PII removal at scale
5. Describe the three-stage training lifecycle: pretraining → SFT → alignment
6. Use extended thinking (O1-style) via the Claude API

---

## Setup Options

This assignment supports three deployment paths. Choose the one that fits your setup:

### Path A: Claude API (Cloud) — Recommended

- **Default model:** `claude-sonnet-4-6`
- **Cost:** ~$1–3 for the full assignment
- **Advantages:** Best quality, extended thinking support, fastest response
- **Requires:** `ANTHROPIC_API_KEY` in your `.env` file

### Path B: Ollama (Local / Free)

- **Default model:** `qwen3.5:27b`
- **Cost:** Free (runs locally)
- **Advantages:** No API key needed, full privacy
- **Requires:** ~20GB RAM, Ollama installed, `ollama pull qwen3.5:27b`

### Path C: Hybrid

- Use both Claude (for quality experiments) and Ollama (for free exploration)
- Best of both worlds for learning

---

## Prerequisites

- Python 3.8+
- Completed Week 1 homework (recommended, not required)
- Internet access (for web scraping and model downloads)

### System Dependencies

These must be installed **separately** from Python packages:

```bash
# macOS (using Homebrew)
brew install tesseract poppler ffmpeg

# Ubuntu / Debian
sudo apt install tesseract-ocr poppler-utils ffmpeg

# Windows
# Tesseract: https://github.com/tesseract-ocr/tesseract/releases
# poppler:   https://github.com/oschwartz10612/poppler-windows/releases
# ffmpeg:    https://ffmpeg.org/download.html
```

---

## Installation

```bash
# 1. Clone and enter the directory
cd Homework2-Submission

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Download spaCy model (required by Presidio PII detection)
python -m spacy download en_core_web_lg

# 5. Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 6. (Path B/C only) Start Ollama and pull the default model
ollama serve
ollama pull qwen3.5:27b

# 7. Launch Jupyter
cd notebooks
jupyter notebook
```

---

## Repository Structure

```
Homework2-Submission/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
├── .gitignore                # Git ignore rules
│
├── notebooks/                # 9 sequential Jupyter notebooks
│   ├── 00_setup_verification.ipynb        # 5 min  — package & env checks
│   ├── 01_environment_setup.ipynb         # 20 min — path selection, first API call
│   ├── 02_transformer_architecture.ipynb  # 35 min — attention, FFN, PE, LayerNorm
│   ├── 03_tokenization.ipynb              # 25 min — BPE, tiktoken, cost analysis
│   ├── 04_pretraining_data_collection.ipynb # 30 min — web scraping, OCR, ASR
│   ├── 05_data_cleaning_pipeline.ipynb    # 30 min — dedup, PII removal, quality
│   ├── 06_finetuning_and_alignment.ipynb  # 20 min — SFT, LoRA, DPO/PPO concepts
│   ├── 07_test_time_scaling.ipynb         # 25 min — CoT, extended thinking, quant
│   └── 08_project_integration.ipynb       # 35 min — apply everything to your project
│
├── src/                      # Reusable helper modules
│   ├── __init__.py           # Package exports
│   ├── config.py             # Path & model configuration
│   ├── llm_client.py         # Unified Claude/Ollama interface
│   ├── cost_tracker.py       # Token & cost tracking
│   ├── utils.py              # Formatting, saving, reflection helpers
│   ├── prompt_templates.py   # CO-STAR framework templates
│   ├── attention_utils.py    # NumPy attention, BertViz, MiniTransformerBlock
│   ├── tokenizer_utils.py    # tiktoken, HF tokenizers, SentencePiece
│   └── data_utils.py         # Scraping, OCR, ASR, cleaning pipeline
│
├── outputs/                  # Auto-generated student deliverables
│   ├── homework_reflection.md     # Primary graded artifact (built incrementally)
│   ├── my_project_update.md       # Week 2 project plan update
│   ├── path_selection.md          # Path A/B/C rationale
│   ├── setup_summary.txt          # Environment summary
│   ├── tokenizer_comparison.json  # Notebook 03 results
│   ├── attention_heatmaps/        # Notebook 02 visualisations
│   ├── arxiv_clean.json           # Notebook 04 web scraping results
│   ├── pdf_ocr/                   # Notebook 04 OCR output texts
│   ├── talks_transcripts.jsonl    # Notebook 04 ASR transcripts
│   ├── clean_corpus.txt           # Notebook 05 cleaned corpus
│   ├── corpus_stats.md            # Notebook 05 cleaning statistics
│   └── thinking_traces/           # Notebook 07 extended thinking JSON
│
└── docs/                     # Additional documentation (optional)
```

---

## Assignment Structure

| Notebook | Topic | Time | Key Deliverable |
|---|---|---|---|
| **00** | Setup Verification | 5 min | — |
| **01** | Environment Setup | 20 min | `path_selection.md` |
| **02** | Transformer Architecture | 35 min | `attention_heatmaps/` |
| **03** | Tokenization | 25 min | `tokenizer_comparison.json` |
| **04** | Pretraining Data Collection | 30 min | `arxiv_clean.json`, `pdf_ocr/`, `talks_transcripts.jsonl` |
| **05** | Data Cleaning Pipeline | 30 min | `clean_corpus.txt`, `corpus_stats.md` |
| **06** | Fine-tuning & Alignment | 20 min | Reflection entries |
| **07** | Test-Time Scaling | 25 min | `thinking_traces/` |
| **08** | Project Integration | 35 min | `my_project_update.md` |

**Total estimated time:** ~3.5 hours

---

## Deliverables (Submit to Canvas)

1. **`outputs/homework_reflection.md`** — Primary graded deliverable
   - Built incrementally as you complete each notebook
   - Should contain reflections from all 8 notebooks (01–08)

2. **`outputs/my_project_update.md`** — Secondary deliverable
   - Updates your Week 1 project definition with Week 2 learnings
   - Must include data strategy, architecture constraints, and updated approach

---

## Grading Rubric

| Component | Weight | Criteria |
|---|---|---|
| **Homework Reflection** | 70% | Depth of insight, evidence of experimentation, clear reasoning |
| **Project Update** | 20% | Clear data strategy, feasible approach, alignment with Week 2 topics |
| **Notebook Completion** | 10% | All notebooks executed, TODOs filled in, no runtime errors |

---

## Cost Estimates

| Path | Model | Estimated Total Cost |
|---|---|---|
| A (Claude) | `claude-sonnet-4-6` | $1–3 |
| B (Ollama) | `qwen3.5:27b` | Free |
| C (Hybrid) | Both | $0.50–1.50 |

Extended thinking (Notebook 07) adds ~$0.50–1.00 for Path A.

---

## Troubleshooting

### Tesseract not found

```bash
# macOS
brew install tesseract

# Verify
tesseract --version
```

### pdf2image fails: "Unable to get page count"

This means poppler is not installed:
```bash
# macOS
brew install poppler

# Ubuntu
sudo apt install poppler-utils
```

### faster-whisper: "No module named 'ctranslate2'"

```bash
pip install faster-whisper --upgrade
```

### BertViz doesn't display in Jupyter

BertViz requires Jupyter widgets. If using VS Code:
```bash
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension
```

If BertViz still fails, the notebook falls back to matplotlib heatmaps automatically.

### Presidio: "No spacy model found"

```bash
python -m spacy download en_core_web_lg
```

If that's too large (~500MB), use the smaller model:
```bash
python -m spacy download en_core_web_sm
```

### Ollama: "model not found"

```bash
ollama pull qwen3.5:27b
# Wait for download to complete (~15GB)
```

### Claude API: "rate limit exceeded"

Wait 60 seconds and retry. The assignment uses small requests, so this should be rare.

### Extended thinking: "thinking parameter not supported"

Make sure `anthropic>=0.40.0` is installed:
```bash
pip install --upgrade anthropic
```

---

## Bonus Challenges (Optional — Extra Credit)

1. **Surya OCR vs Tesseract** (5%) — Install `surya-ocr` and compare quality on the same PDFs
2. **Extended thinking cost analysis** (3%) — Plot accuracy vs budget_tokens for 5+ values
3. **Deduplication visualizer** (3%) — Create a matplotlib graph showing the similarity matrix
4. **Multilingual tokenization** (3%) — Compare tokenizer efficiency on 5+ languages

---

## Resources

### Transformer Architecture
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. 2017
- [Dive into Deep Learning: Transformers](https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html)

### Tokenization
- [BPE explained](https://huggingface.co/learn/nlp-course/chapter6/5) — HuggingFace NLP Course
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI tokenizer

### Data Pipeline Tools
- [Trafilatura](https://github.com/adbar/trafilatura) — Web content extraction
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Fast Whisper ASR
- [Presidio](https://github.com/microsoft/presidio) — PII detection & anonymization
- [datasketch](https://datasketch.readthedocs.io/) — MinHash deduplication

### Fine-Tuning & Alignment
- [LoRA paper](https://arxiv.org/abs/2106.09685) — Hu et al. 2021
- [DPO paper](https://arxiv.org/abs/2305.18290) — Rafailov et al. 2023
- [HuggingFace PEFT](https://huggingface.co/docs/peft/) — Parameter-efficient fine-tuning

### Extended Thinking
- [Claude Extended Thinking Docs](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) — Wei et al. 2022

---

## Timeline (Suggested 7-Day Schedule)

| Day | Notebooks | Focus |
|---|---|---|
| 1 | 00, 01 | Setup and orientation |
| 2 | 02 | Transformer architecture deep-dive |
| 3 | 03 | Tokenization experiments |
| 4 | 04 | Data collection (web, PDF, audio) |
| 5 | 05 | Data cleaning pipeline |
| 6 | 06, 07 | Fine-tuning concepts + test-time scaling |
| 7 | 08 | Project integration + polish reflection |

---

## Support

- **Discord:** Post questions in the #week-2 channel
- **Office Hours:** Check the course calendar
- **Issues:** If you find a bug in the notebooks, report it to the instructor

---

*Happy learning!* 🚀
