"""Print the training arguments a run actually used, read from its TensorBoard hparams.

`lotus/train_iris_ms2_g.py` passes `vars(args)` to `accelerator.init_trackers`, so
the full argument set lives in the run's event files even when the job's stdout is
gone. Given several runs, prints only the arguments that differ between them (plus
a few that always matter), so "these two arms differ in exactly one switch" can be
checked instead of assumed.

    python tools/dump_run_args.py <run_dir> [<run_dir> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

ALWAYS = ("seed", "norm_type", "train_data_dir_ms2", "ms2_manifest", "max_train_steps",
          "pretrained_model_name_or_path", "learning_rate", "train_batch_size",
          "gradient_accumulation_steps", "lambda_image", "no_captions")


def hparams(run: Path) -> dict:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.plugins.hparams import plugin_data_pb2

    found: dict = {}
    for ev_file in sorted(run.rglob("events.out.tfevents*")):
        try:
            for event in EventFileLoader(str(ev_file)).Load():
                for value in event.summary.value:
                    meta = value.metadata.plugin_data
                    if meta.plugin_name != "hparams":
                        continue
                    data = plugin_data_pb2.HParamsPluginData.FromString(meta.content)
                    if data.HasField("session_start_info"):
                        for k, v in data.session_start_info.hparams.items():
                            kind = v.WhichOneof("kind")
                            found[k] = getattr(v, kind) if kind else None
        except Exception as exc:  # unreadable file: say so, keep going
            print(f"!! {ev_file}: {exc}", file=sys.stderr)
    return found


def main() -> None:
    runs = [Path(p) for p in sys.argv[1:]]
    table = {r.name: hparams(r) for r in runs}
    for name, h in table.items():
        print(f"[{name}] {len(h)} hparams" + ("" if h else "  !! none found"))
    keys = sorted(set().union(*table.values()))
    show = [k for k in keys if k in ALWAYS or len({str(h.get(k)) for h in table.values()}) > 1]
    width = max([len(k) for k in show] + [8])
    print("\n" + "arg".ljust(width) + "".join(f"  {n[:34]:34s}" for n in table))
    for k in show:
        print(k.ljust(width) + "".join(f"  {str(h.get(k))[:34]:34s}" for h in table.values()))
    # Environment-only switches (thermal stretch) are not argparse arguments.
    print("\nnote: IRIS_THERMAL_STRETCH is an environment variable, not an argument; "
          "it is read from the eval JSON instead.")


if __name__ == "__main__":
    main()
