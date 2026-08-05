# Data layout (not shipped in this repository)

This GitHub repository is **code-only**. Large weights, embeddings, dictionary
artifacts, and the painting catalogue are omitted.

## Expected tree after you obtain artifacts

```text
data/
  examples/                 # small demo images (tracked in git)
  models/
    VHM-B-16.pth            # student ViT weights
    student_head.pth        # Z-embed head
  embeddings/
    local_patch_pooled.npy
    local_patch_pooled_inference_manifest.json
  artifacts/
    global_ksvd/            # dictionary, mean, config, sparse codes, meta
    atom_descriptions/      # Formal Evidence atom JSON (+ optional caches)
  Painting_Databaseb.xlsx   # catalogue spreadsheet
  outputs/                  # created at runtime (gitignored)
```

## How to obtain

1. **Colab zip** (full runtime bundle): build with `bash scripts/pack_colab_zip.sh`
   on a machine that already has the artifacts, then host the zip privately, **or**
2. Place files locally under `data/` as above, then:

```bash
pip install -e .
export VISUAL_HISTORY_AGENT_ROOT=$PWD
# LLM keys in project `.env` (see `.env.example`)
```

Rebuild dictionary / Formal Evidence only if you have the full embedding corpus:

```bash
python scripts/train_global_dictionary.py
python scripts/run_formal_evidence_agent.py --rcs-only
python scripts/run_formal_evidence_agent.py --synthesis-only
```
