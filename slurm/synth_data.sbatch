#!/bin/bash
#SBATCH --job-name=iris_synth
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
# Synthetic training data for the Iris-style pretraining (docs/IRIS_SYNTH_PRETRAIN_V1.md).
#   STEP=probe              one Hypersim scene + vKITTI headers, environment checks
#   STEP=model              download InternVL3-8B into $HF_HOME (caption env)
#   STEP=hypersim           array task, COUNT=<total shards>
#   STEP=hypersim-finalize  every scene fetched; metadata.jsonl
#   STEP=vkitti             download rgb/depth/textgt, extract, check layout, delete tars
#   STEP=vkitti-normals     (GPU) Lotus depth2normal for the four training scenes, then check
#   STEP=caption-finalize   COUNT=<caption shards>; merge into Iris's two caption JSON files
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
SYN="${SYNTH_ROOT:-$SCRATCH/data/synth}"
HY="$SYN/hypersim" VK="$SYN/vkitti" TARS="$SYN/vkitti_tars"
VK_URL=https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_
PARTS=(rgb.tar depth.tar textgt.tar.gz)

case "${STEP:?STEP=probe|sd20|model|hypersim|hypersim-finalize|vkitti|vkitti-normals}" in
probe)
  python tests/test_synth_data.py
  python -u tools/synth_data.py probe --out "$HY"
  if conda env list | grep -E '^caption ' >/dev/null; then echo "[probe] conda env caption: present"; else echo "[probe] conda env caption: MISSING"; fi
  if [[ -d "$HF_HOME/hub/models--OpenGVLab--InternVL3-8B" ]]; then echo "[probe] InternVL3-8B: cached"; else echo "[probe] InternVL3-8B: NOT in $HF_HOME"; fi
  lfs quota -h -u "$USER" /scratch | tail -2
  ;;
sd20)
  # SD 2.0-base, the starting point Iris's train_iris_g_depth.sh names
  # (stabilityai/stable-diffusion-2-base, unreachable: see iris_ms2.sbatch). Three unrelated
  # mirrors (Manojb, PeggyWang, luongphamit) carry the same U-Net safetensors, sha256 below,
  # plus the official 512-base-ema.safetensors; VAE and text encoder are byte-identical to
  # the 2.1-base mirror every sd2 arm uses, so only the U-Net differs.
  python - <<'PY'
import hashlib, os
from huggingface_hub import snapshot_download
p = snapshot_download('Manojb/stable-diffusion-2-base',  # main was 64bf7b4f on 2026-11-14
                      allow_patterns=['*.json', '*.txt', '*/model.safetensors', '*/diffusion_pytorch_model.safetensors'])
h = hashlib.sha256()
with open(os.path.join(p, 'unet', 'diffusion_pytorch_model.safetensors'), 'rb') as f:
    for b in iter(lambda: f.read(1 << 24), b''):
        h.update(b)
want = 'c60b3631cc85148ca0cc524daff7767efb707fd26c40d46cc3cd217dc5252cd0'
print(p, h.hexdigest())
assert h.hexdigest() == want, 'U-Net sha256 mismatch'
PY
  ;;
model)
  # InternVL3-8B into $HF_HOME (about 16 GB); synth_caption.sbatch then runs offline.
  conda activate caption
  python -c "from huggingface_hub import snapshot_download as d; print(d('OpenGVLab/InternVL3-8B'))"
  ;;
hypersim)
  python -u tools/synth_data.py hypersim --out "$HY" --index "${SLURM_ARRAY_TASK_ID:-0}" --count "${COUNT:?}"
  ;;
hypersim-finalize)
  python -u tools/synth_data.py hypersim-finalize --out "$HY"
  ;;
vkitti)
  mkdir -p "$TARS" "$VK"
  for part in "${PARTS[@]}"; do
    curl -fsSL --retry 10 --retry-delay 30 -C - -o "$TARS/$part" "$VK_URL$part"
    echo "[vkitti] $part $(du -h "$TARS/$part" | cut -f1)"
  done
  # Each archive is extracted on its own, then hard-linked into one tree: the archives may or
  # may not carry a top-level folder, and the Scene directories must merge.
  for part in "${PARTS[@]}"; do
    stage="$SYN/stage_${part%%.*}"
    rm -rf "$stage" && mkdir -p "$stage"
    tar -xf "$TARS/$part" -C "$stage"
    top=$(dirname "$(find "$stage" -maxdepth 3 -type d -name Scene02 | head -n 1)")
    [[ -d "$top/Scene02" ]] || { echo "!! no Scene02 in $part"; exit 1; }
    cp -rl "$top/." "$VK/"
    rm -rf "$stage"
  done
  python -u tools/synth_data.py vkitti-check --out "$VK"
  rm -rf "$TARS"
  ;;
vkitti-normals)
  # NORMAL_SCENES narrows the pass, e.g. to redo the scene a timed-out run was in
  # (depth2normal walks the scenes in order and overwrites what it writes). The
  # check below always covers all four.
  (cd lotus && PYTHONPATH="$PWD" python -u utils/depth2normal.py --data_path "$VK" --batch_size 10 --scenes ${NORMAL_SCENES:-02 06 18 20})
  python -u tools/synth_data.py vkitti-check --out "$VK" --normals
  ;;
caption-finalize)
  python tests/test_caption_synth.py
  python -u tools/caption_synth.py finalize --hypersim "$HY" --vkitti "$VK" --out "$SYN/captions" --count "${COUNT:?}"
  ;;
*) echo "!! unknown STEP=$STEP"; exit 1 ;;
esac
