#!/usr/bin/env bash
# Build a self-contained Colab distribution zip (no Mac-local paths required at runtime).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$ROOT")"
OUT="${1:-$PARENT/Visual_History_Agent.zip}"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/vha_pack.XXXXXX")"
NAME="Visual_History_Agent"

cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT

mkdir -p "$STAGE/$NAME"

echo "Staging code…"
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '.ipynb_checkpoints' \
  "$ROOT/visual_history_agent/" "$STAGE/$NAME/visual_history_agent/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  "$ROOT/scripts/" "$STAGE/$NAME/scripts/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  "$ROOT/vendor/" "$STAGE/$NAME/vendor/"

cp "$ROOT/pyproject.toml" "$STAGE/$NAME/"
cp "$ROOT/requirements-colab.txt" "$STAGE/$NAME/"
cp "$ROOT/README.md" "$STAGE/$NAME/" 2>/dev/null || true
cp "$ROOT/.env.example" "$STAGE/$NAME/" 2>/dev/null || true

copy_file() {
  local rel="$1"
  mkdir -p "$STAGE/$NAME/$(dirname "$rel")"
  cp "$ROOT/$rel" "$STAGE/$NAME/$rel"
}

echo "Staging required data artifacts…"
REQUIRED=(
  data/models/VHM-B-16.pth
  data/models/student_head.pth
  data/artifacts/global_ksvd/global_patch_pooled_dictionary_atoms.npy
  data/artifacts/global_ksvd/global_patch_pooled_embedding_mean.npy
  data/artifacts/global_ksvd/global_patch_pooled_ksvd_config.json
  data/artifacts/global_ksvd/global_patch_pooled_sparse_codes.npy
  data/artifacts/global_ksvd/global_dictionary_meta.json
  data/artifacts/atom_descriptions/global_atom_painterly_descriptions.json
  data/embeddings/local_patch_pooled.npy
  data/Painting_Databaseb.xlsx
  data/examples/The_Blue_Room.jpeg
  data/examples/Norton_Jerome.jpg
  data/examples/Princeton_Jerome.jpg
)
for rel in "${REQUIRED[@]}"; do
  if [[ ! -f "$ROOT/$rel" ]]; then
    echo "MISSING required file: $ROOT/$rel" >&2
    exit 1
  fi
  copy_file "$rel"
done

mkdir -p "$STAGE/$NAME/data/outputs"

# Sanity: vendor gateway present
if [[ ! -f "$STAGE/$NAME/vendor/VisualHistory/llm.py" ]]; then
  echo "MISSING vendor/VisualHistory/llm.py" >&2
  exit 1
fi
if [[ ! -f "$STAGE/$NAME/scripts/infer_single_image_z_embed.py" ]]; then
  echo "MISSING scripts/infer_single_image_z_embed.py" >&2
  exit 1
fi
if [[ ! -f "$STAGE/$NAME/scripts/run_description_agent.py" ]]; then
  echo "MISSING scripts/run_description_agent.py" >&2
  exit 1
fi

# Manifest for debugging
{
  echo "Visual_History_Agent zip"
  echo "built: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "VISUALHISTORY_ROOT inside zip: vendor/  (parent of VisualHistory/)"
  echo
  echo "required files:"
  for rel in "${REQUIRED[@]}"; do
    echo "  $rel"
  done
} > "$STAGE/$NAME/MANIFEST.txt"

echo "Zipping…"
rm -f "$OUT"
(cd "$STAGE" && zip -r -q "$OUT" "$NAME")
ls -lh "$OUT"

python3 - <<PY
from pathlib import Path
import zipfile
z = zipfile.ZipFile("$OUT")
names = set(z.namelist())
need = [
    "Visual_History_Agent/vendor/VisualHistory/llm.py",
    "Visual_History_Agent/scripts/run_description_agent.py",
    "Visual_History_Agent/scripts/infer_single_image_z_embed.py",
    "Visual_History_Agent/requirements-colab.txt",
    "Visual_History_Agent/pyproject.toml",
] + [f"Visual_History_Agent/{r}" for r in """${REQUIRED[@]}""".split()]
missing = [n for n in need if n not in names]
if missing:
    raise SystemExit("zip missing:\\n  " + "\\n  ".join(missing))
print(f"OK: {len(names)} members; vendor + scripts + required data present")
PY

echo
echo "Upload/replace on Drive → Anyone with the link → Viewer → set DRIVE_ZIP_FILE_ID in Colab."
