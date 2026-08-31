"""Wait for Marigold's day run to show a real rate, then submit night and rain.

Two things have to be true before eight more jobs are worth submitting, and
neither can be known at submission time:

  1. The exam matches the training normalisation. Marigold trains under
     truncnorm, whose output is normalised depth; read as disparity it scores
     AbsRel 0.265 with delta1 0.504, which looks like a broken model. The val
     number is the cheapest place to see that, so it gates everything below.
  2. The 50-step rate. One step measured 0.112 s/frame; how much of that is the
     U-Net -- hence what 50 steps costs -- was never measured, and the honest
     range spanned 1.5 to 5.6 s/frame. A walltime picked from that range is a
     guess. Read from the progress bar it is a number.

Run it after the day jobs are running. It blocks until the bar is stable, then
submits, so it can be left alone.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path(os.environ["IRIS_RUNS"])
MANIFESTS = Path(os.environ["IRIS_MANIFEST_DIR"])
LOGS = Path(os.environ.get("SLURM_LOG_DIR", Path(os.environ["SCRATCH"]) / "logs"))
REPO = Path(os.environ.get("IRIS_REPO", Path.home() / "Iris"))

DAY_FRAMES = 23311
VAL_CEILING = 0.15          # anything above this is a mismatched exam, not a weak model
MIN_ITERS = 200             # tqdm's rate is noise before this
CONDITIONS = {"night": ("ms2_test_night3_thermalcap_20260821.jsonl", 22916),
              "rain":  ("ms2_test_rainy3_thermalcap_20260821.jsonl", 25023)}
ARMS = [("mth_c", "marigold_full8_thermalcap", "correct", "correct"),
        ("mth_e", "marigold_full8_thermalcap", "correct", "empty"),
        ("mnc_e", "marigold_full8_nocap", "empty", "empty"),
        ("mnc_c", "marigold_full8_nocap", "empty", "correct")]

BAR = re.compile(r"(\d+)/" + str(DAY_FRAMES) + r" \[[^\]]*?([\d.]+)(s/it|it/s)\]")


def newest(pattern):
    hits = sorted(LOGS.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def val_abs_rel():
    """The val AbsRel of whichever Marigold arm has written one."""
    for path in RUNS.glob("eval/marigold_full8_*_val_*_s20000/eval_eval*.json"):
        try:
            return path, json.loads(path.read_text())["abs_rel"]
        except Exception:
            continue
    return None, None


def rate_seconds_per_frame():
    log = newest("mth_c-*.err")
    if log is None:
        return None, "no mth_c log yet"
    text = log.read_bytes().decode("utf-8", "replace").replace("\r", "\n")
    best = None
    for done, value, unit in BAR.findall(text):
        if int(done) < MIN_ITERS:
            continue
        best = float(value) if unit == "s/it" else 1.0 / float(value)
    if best is None:
        return None, f"{log.name}: bar not past {MIN_ITERS} frames yet"
    return best, log.name


def already_queued(condition):
    """Names of this condition's arms that squeue already knows about.

    Without this, running the script twice submits the eight jobs twice --
    likely, because whether a nohup process survives logout depends on the
    login node's logind policy, so the honest advice is "check, and re-run if
    it died", and that advice is only safe if re-running is safe.
    """
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j"],
                         capture_output=True, text=True)
    live = set(out.stdout.split())
    return [f"{name}_{condition}" for name, *_ in ARMS if f"{name}_{condition}" in live]


def submit(condition, manifest, frames, seconds_per_frame):
    clash = already_queued(condition)
    if clash:
        print(f"  {condition}: 跳过，{' '.join(clash)} 已在队列里")
        return
    # 30% headroom over the measured rate, then a whole hour for load and the
    # short val pass. Slurm cannot raise a running job's limit, so the cost of
    # being low is the whole pass; the cost of being high is queue position.
    hours = int(frames * seconds_per_frame * 1.3 / 3600) + 1
    ids = []
    for name, run, sel, prompt in ARMS:
        env = dict(os.environ,
                   RUN=run, STEPS="20000", SEL_PROMPT=sel, TEST_PROMPTS=prompt,
                   STEPS_INFER="50", VAL_STRIDE="200", ALIGN_MODE="ssi",
                   LOTUS_MODEL="prs-eth/marigold-v1-0", BACKBONE="marigold",
                   VAL_MANIFEST=str(MANIFESTS / "ms2_val_official3_thermalcap_20260821.jsonl"),
                   TEST_MANIFEST=str(MANIFESTS / manifest))
        keys = ("RUN,STEPS,SEL_PROMPT,TEST_PROMPTS,STEPS_INFER,VAL_STRIDE,ALIGN_MODE,"
                "LOTUS_MODEL,BACKBONE,VAL_MANIFEST,TEST_MANIFEST")
        out = subprocess.run(
            ["sbatch", "--parsable", "-J", f"{name}_{condition}",
             f"--time={hours:02d}:00:00", f"--export=ALL,{keys}",
             str(REPO / "slurm" / "iris_ms2_pipeline.sbatch")],
            env=env, capture_output=True, text=True, check=True)
        ids.append(out.stdout.strip())
    print(f"  {condition}: {frames} frames, {hours}h walltime -> {' '.join(ids)}")


def main():
    path, abs_rel = val_abs_rel()
    while abs_rel is None:
        print("等 val 结果...", flush=True)
        time.sleep(60)
        path, abs_rel = val_abs_rel()
    print(f"val AbsRel = {abs_rel:.5f}   ({path.parent.name})")
    if abs_rel > VAL_CEILING:
        sys.exit(f"⛔ val AbsRel {abs_rel:.5f} > {VAL_CEILING}. 这不是模型弱，是卷子还没对上。\n"
                 f"   先看 {path} 旁边日志里的『对齐』那一行，别提交 night/rain。")
    print(f"✅ 低于 {VAL_CEILING}，对齐是对的。")

    rate, note = rate_seconds_per_frame()
    while rate is None:
        print(f"等进度条... ({note})", flush=True)
        time.sleep(120)
        rate, note = rate_seconds_per_frame()
    print(f"实测 {rate:.3f} s/帧   ({note})")
    print(f"day 单条 pass 推算 {DAY_FRAMES * rate / 3600:.1f} 小时")
    print("提交：")
    for condition, (manifest, frames) in CONDITIONS.items():
        submit(condition, manifest, frames, rate)


if __name__ == "__main__":
    main()
