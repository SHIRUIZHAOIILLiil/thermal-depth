"""Dump the full network structure of every route in the 20-epoch suite.

Task 3 of the 2026-07-25 plan: for each route, print the complete module tree
with per-module parameter counts and an explicit trained / frozen flag, the way
the Iris paper reports its architecture.

The six routes (a/b are the baselines; c/d each come in an adapter-only and a
joint adapter+U-Net variant):

    a       RGB     -> frozen VAE encode -> TRAIN U-Net -> frozen VAE decode
    b       Thermal -> frozen VAE encode -> TRAIN U-Net -> frozen VAE decode
    c1      Thermal -> frozen VAE encode -> TRAIN Adapter -> frozen U-Net
    c2      Thermal -> frozen VAE encode -> TRAIN Adapter -> TRAIN U-Net
    d1      Thermal -> frozen AnyThermal -> TRAIN Adapter -> frozen U-Net
    d2      Thermal -> frozen AnyThermal -> TRAIN Adapter -> TRAIN U-Net

Everything runs on CPU in fp32 and only counts parameters -- no GPU, no data,
no forward pass.  Usage:

    python tools/dump_route_architecture.py --output docs/ROUTE_ARCHITECTURES_20EPOCH.md

Add ``--skip-anythermal`` if the AnyThermal weights are not cached locally; the
d-route tables are then emitted without the encoder row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# --------------------------------------------------------------------------- #
# route definitions
# --------------------------------------------------------------------------- #

# name -> (input modality, condition source, trains adapter, trains U-Net, blurb)
ROUTES = {
    "a_rgb_unet": {
        "title": "线 a — RGB 输入，只训 U-Net（Baseline）",
        "modality": "rgb",
        "condition": "vae",
        "train_adapter": False,
        "train_unet": True,
    },
    "b_thermal_unet": {
        "title": "线 b — Thermal 输入，只训 U-Net（Baseline）",
        "modality": "thermal",
        "condition": "vae",
        "train_adapter": False,
        "train_unet": True,
    },
    "c1_vae_adapter": {
        "title": "线 c1 — Thermal，VAE 后接 Adapter，U-Net 冻结",
        "modality": "thermal",
        "condition": "vae_adapter",
        "train_adapter": True,
        "train_unet": False,
    },
    "c2_vae_adapter_unet": {
        "title": "线 c2 — Thermal，VAE 后接 Adapter，Adapter + U-Net 联合训练",
        "modality": "thermal",
        "condition": "vae_adapter",
        "train_adapter": True,
        "train_unet": True,
    },
    "d1_anythermal_adapter": {
        "title": "线 d1 — Thermal → AnyThermal → Adapter，U-Net 冻结",
        "modality": "thermal",
        "condition": "anythermal_adapter",
        "train_adapter": True,
        "train_unet": False,
    },
    "d2_anythermal_adapter_unet": {
        "title": "线 d2 — Thermal → AnyThermal → Adapter，Adapter + U-Net 联合训练",
        "modality": "thermal",
        "condition": "anythermal_adapter",
        "train_adapter": True,
        "train_unet": True,
    },
}

ROLE_TEXT = {
    "vae_encoder": "把输入图编码成 4 通道 latent，作为 U-Net 的 condition",
    "vae_decoder": "把 U-Net 输出的 latent 解回视差图（GT 损失在这之后算）",
    "vae_quant": "编码后的 latent 量化卷积（encoder 侧）",
    "vae_post_quant": "解码前的 latent 反量化卷积（decoder 侧）",
    "text_encoder": "把 prompt（无 caption 阶段恒为空串）编码成 cross-attention 条件",
    "unet": "Lotus 主干，单步 x0 预测",
    "adapter_vae": "在冻结 VAE latent 上的残差 CNN，零初始化 ⇒ 未训练时是恒等",
    "adapter_anythermal": "把 AnyThermal token 转成 U-Net 可读的 4 通道 latent",
    "anythermal": "AnyThermal 热像基础模型（DINOv2 主干），提供热像特征",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "ROUTE_ARCHITECTURES_20EPOCH.md",
        help="Markdown report path. A sibling .json with the same stem is also written.",
    )
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-anythermal",
        action="store_true",
        help="Do not load the AnyThermal encoder (d-route tables omit its row).",
    )
    parser.add_argument("--adapter-hidden-channels", type=int, default=256)
    parser.add_argument("--adapter-blocks", type=int, default=6)
    parser.add_argument(
        "--expand-depth",
        type=int,
        default=1,
        help="How many nn.Module levels to expand inside each top-level component.",
    )
    parser.add_argument(
        "--routes",
        nargs="*",
        default=None,
        help=f"Subset of routes to dump. Default: all of {list(ROUTES)}",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# parameter accounting
# --------------------------------------------------------------------------- #


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def human(count: int) -> str:
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f} B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.2f} M"
    if count >= 1_000:
        return f"{count / 1_000:.2f} K"
    return str(count)


def child_rows(module: nn.Module, depth: int, prefix: str = "") -> Iterable[dict]:
    """Yield one row per child module down to ``depth`` levels."""
    if depth <= 0:
        return
    for name, child in module.named_children():
        total = count_parameters(child)
        if total == 0:
            continue
        yield {
            "name": f"{prefix}{name}",
            "class": type(child).__name__,
            "parameters": total,
        }
        yield from child_rows(child, depth - 1, prefix=f"{prefix}{name}.")


def component_row(label: str, module: nn.Module, trainable: bool, role: str) -> dict:
    total = count_parameters(module)
    return {
        "component": label,
        "class": type(module).__name__,
        "parameters": total,
        "trainable": trainable,
        "role": role,
    }


# --------------------------------------------------------------------------- #
# model construction (mirrors the trainers)
# --------------------------------------------------------------------------- #


def build_shared(args: argparse.Namespace) -> dict:
    """Load Lotus once and the two adapter classes; reused across all routes."""
    from pipeline import LotusGPipeline  # noqa: E402  (needs lotus/ on sys.path)

    from models.thermal_vae_latent_adapter import ThermalVAELatentAdapter  # noqa: E402
    from models.anythermal_lotus_adapter_v2_3 import AnyThermalLotusAdapterV23  # noqa: E402

    print(f"[load] Lotus pipeline: {args.lotus_model_path} (cpu, fp32)", flush=True)
    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    )

    shared = {
        "vae": lotus.vae,
        "text_encoder": lotus.text_encoder,
        "unet": lotus.unet,
        "vae_latent_adapter": ThermalVAELatentAdapter(
            hidden_channels=args.adapter_hidden_channels, blocks=args.adapter_blocks
        ),
        "anythermal_adapter": AnyThermalLotusAdapterV23(),
        "anythermal": None,
    }

    if not args.skip_anythermal:
        from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402

        print(f"[load] AnyThermal encoder: {args.anythermal_model_path} (cpu)", flush=True)
        encoder = AnyThermalEncoder(
            model_path=args.anythermal_model_path,
            device="cpu",
            local_files_only=args.local_files_only,
        )
        shared["anythermal"] = encoder.model

    return shared


def route_components(route: dict, shared: dict) -> list[dict]:
    """Ordered list of the modules a route actually instantiates."""
    vae = shared["vae"]
    rows: list[dict] = []

    if route["condition"] in ("vae", "vae_adapter"):
        rows.append(component_row("VAE encoder", vae.encoder, False, ROLE_TEXT["vae_encoder"]))
        rows.append(component_row("VAE quant_conv", vae.quant_conv, False, ROLE_TEXT["vae_quant"]))
    else:
        if shared["anythermal"] is not None:
            rows.append(
                component_row("AnyThermal encoder", shared["anythermal"], False, ROLE_TEXT["anythermal"])
            )

    if route["condition"] == "vae_adapter":
        rows.append(
            component_row(
                "Adapter (VAE latent)",
                shared["vae_latent_adapter"],
                route["train_adapter"],
                ROLE_TEXT["adapter_vae"],
            )
        )
    elif route["condition"] == "anythermal_adapter":
        rows.append(
            component_row(
                "Adapter (AnyThermal→latent)",
                shared["anythermal_adapter"],
                route["train_adapter"],
                ROLE_TEXT["adapter_anythermal"],
            )
        )

    rows.append(component_row("Text encoder", shared["text_encoder"], False, ROLE_TEXT["text_encoder"]))
    rows.append(component_row("U-Net", shared["unet"], route["train_unet"], ROLE_TEXT["unet"]))
    rows.append(
        component_row("VAE post_quant_conv", vae.post_quant_conv, False, ROLE_TEXT["vae_post_quant"])
    )
    rows.append(component_row("VAE decoder", vae.decoder, False, ROLE_TEXT["vae_decoder"]))
    return rows


def route_module_for(component: str, shared: dict):
    return {
        "VAE encoder": shared["vae"].encoder,
        "VAE decoder": shared["vae"].decoder,
        "Adapter (VAE latent)": shared["vae_latent_adapter"],
        "Adapter (AnyThermal→latent)": shared["anythermal_adapter"],
        "U-Net": shared["unet"],
        "Text encoder": shared["text_encoder"],
        "AnyThermal encoder": shared["anythermal"],
    }.get(component)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def render(routes: dict, shared: dict, args: argparse.Namespace) -> tuple[str, dict]:
    lines: list[str] = []
    payload: dict = {"routes": {}}

    lines.append("# 六条路线的完整网络结构与参数量")
    lines.append("")
    lines.append(
        "由 `tools/dump_route_architecture.py` 自动生成。参数量为 fp32 权重的元素个数，"
        "「状态」列即训练时 `requires_grad` 的取值。"
    )
    lines.append("")

    # ---- cross-route summary ------------------------------------------------
    lines.append("## 0. 六条线总览")
    lines.append("")
    lines.append("| 线 | 输入 | Condition 来源 | 可训练模块 | 可训练参数 | 冻结参数 | 可训练占比 |")
    lines.append("|---|---|---|---|---:|---:|---:|")

    summary = {}
    for key, route in routes.items():
        rows = route_components(route, shared)
        trainable = sum(row["parameters"] for row in rows if row["trainable"])
        frozen = sum(row["parameters"] for row in rows if not row["trainable"])
        trained_names = [row["component"] for row in rows if row["trainable"]] or ["（无）"]
        summary[key] = {"rows": rows, "trainable": trainable, "frozen": frozen}
        condition = {
            "vae": "冻结 VAE latent",
            "vae_adapter": "VAE latent + Adapter",
            "anythermal_adapter": "AnyThermal 特征 + Adapter",
        }[route["condition"]]
        share = trainable / (trainable + frozen) if trainable + frozen else 0.0
        lines.append(
            f"| {key} | {'RGB' if route['modality'] == 'rgb' else 'Thermal'} | {condition} | "
            f"{' + '.join(trained_names)} | {human(trainable)} | {human(frozen)} | {share:.1%} |"
        )
    lines.append("")

    # ---- per-route detail ---------------------------------------------------
    for index, (key, route) in enumerate(routes.items(), start=1):
        rows = summary[key]["rows"]
        lines.append(f"## {index}. {route['title']}")
        lines.append("")
        lines.append("### 前向顺序与冻结状态")
        lines.append("")
        lines.append("| # | 模块 | 类 | 参数量 | 状态 | 作用 |")
        lines.append("|---:|---|---|---:|---|---|")
        for order, row in enumerate(rows, start=1):
            state = "**训练**" if row["trainable"] else "冻结"
            lines.append(
                f"| {order} | {row['component']} | `{row['class']}` | "
                f"{row['parameters']:,} | {state} | {row['role']} |"
            )
        lines.append(
            f"| | **合计** | | **{summary[key]['trainable'] + summary[key]['frozen']:,}** | "
            f"训练 {human(summary[key]['trainable'])} / 冻结 {human(summary[key]['frozen'])} | |"
        )
        lines.append("")

        # expanded sub-tree for the trainable modules only (that is what matters)
        trainable_rows = [row for row in rows if row["trainable"]]
        if trainable_rows and args.expand_depth > 0:
            lines.append("### 可训练模块内部结构")
            lines.append("")
            for row in trainable_rows:
                module = route_module_for(row["component"], shared)
                if module is None:
                    continue
                lines.append(f"**{row['component']}** (`{row['class']}`, {human(row['parameters'])})")
                lines.append("")
                lines.append("| 子模块 | 类 | 参数量 |")
                lines.append("|---|---|---:|")
                for child in child_rows(module, args.expand_depth):
                    lines.append(
                        f"| `{child['name']}` | `{child['class']}` | {child['parameters']:,} |"
                    )
                lines.append("")

        payload["routes"][key] = {
            "title": route["title"],
            "modality": route["modality"],
            "condition": route["condition"],
            "components": rows,
            "trainable_parameters": summary[key]["trainable"],
            "frozen_parameters": summary[key]["frozen"],
        }

    return "\n".join(lines) + "\n", payload


def main() -> None:
    args = parse_args()
    selected = args.routes or list(ROUTES)
    unknown = [name for name in selected if name not in ROUTES]
    if unknown:
        raise SystemExit(f"Unknown routes: {unknown}; valid names are {list(ROUTES)}")
    routes = {name: ROUTES[name] for name in selected}

    needs_anythermal = any(
        ROUTES[name]["condition"] == "anythermal_adapter" for name in selected
    )
    if needs_anythermal and args.skip_anythermal:
        print("[warn] --skip-anythermal: d-route tables will omit the encoder row.", flush=True)

    shared = build_shared(args)
    report, payload = render(routes, shared, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] markdown -> {args.output}")
    print(f"[done] json     -> {json_path}")
    for key, entry in payload["routes"].items():
        print(
            f"  {key:28s} trainable {human(entry['trainable_parameters']):>9s}"
            f"   frozen {human(entry['frozen_parameters']):>9s}"
        )


if __name__ == "__main__":
    main()
