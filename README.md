# Visual History Agent

Code for a painting interpretability pipeline: encode an image, retrieve Formal
Evidence atom labels, produce a cited formal description (Description Agent),
compare two paintings (same-hand then copy), and browse local-Dai spatial atom maps.

This repository is **code-only**. Model weights, embeddings, K-SVD artifacts, and
the catalogue database are not included — see [`data/README.md`](data/README.md).

## Layout

| Path | Role |
|------|------|
| `visual_history_agent/` | Python package (describe, compare, encode, spatial browser) |
| `scripts/` | Training / agents / LaTeX supplements / Colab packer |
| `vendor/VisualHistory/` | Slim vendored LLM gateway (OpenAI + optional Azure) |
| `notebooks/` | Local notebooks `00`–`04` and Colab `colab_02_04.ipynb` |
| `data/examples/` | Small demo images |

## Install

```bash
git clone <this-repo>
cd Visual_History_Agent
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # set OPENAI_API_KEY (and optional Azure keys)
export VISUAL_HISTORY_AGENT_ROOT=$PWD
```

Place required files under `data/` (see [`data/README.md`](data/README.md)), **or**
use the Colab zip workflow below.

## CLI

```bash
python -m visual_history_agent describe --image data/examples/The_Blue_Room.jpeg
python -m visual_history_agent map --image data/examples/The_Blue_Room.jpeg
python -m visual_history_agent compare \
  --a data/examples/Norton_Jerome.jpg \
  --b data/examples/Princeton_Jerome.jpg
```

## Notebooks

1. `00_dictionary_learning.ipynb` — global K-SVD dictionary  
2. `01_formal_evidence_agent.ipynb` — atom evidence (Qwen RCS + synthesis)  
3. `02_single_description.ipynb` — Description Agent  
4. `03_comparison.ipynb` — same-hand then copy  
5. `04_coefficient_maps.ipynb` — spatial atom browser  
6. `colab_02_04.ipynb` — Colab runner for 02→04 via one public zip (`gdown`)

## Colab distribution (optional)

On a machine that already has full `data/` artifacts:

```bash
bash scripts/pack_colab_zip.sh
# → ../Visual_History_Agent.zip
```

Host the zip (e.g. Drive, Anyone with the link → Viewer), put the file id in the
Colab config cell, and set `OPENAI_API_KEY` via the Colab UI or a secret — never
commit keys.

## Agents

| Agent | Role |
|-------|------|
| **Visual History Agent** | Encode, describe, map, compare |
| **Formal Evidence Agent** | Atom-level RCS + synthesis labels |
| **Description Agent** | ReAct formal description grounded in atoms |

## License

MIT — see [`LICENSE`](LICENSE).
