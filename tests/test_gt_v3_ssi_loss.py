"""CPU unit checks for the P1 masked SSI loss (no GPU/dataset needed)."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from train_ms2_joint_gt_v3 import masked_ssi_l1  # noqa: E402


def main() -> None:
    torch.manual_seed(0)
    mask = torch.zeros(1, 64, 64)
    mask[:, 10:50, 10:50] = 1

    # Perfect affine relation: aligned loss must vanish.
    pred = torch.rand(1, 64, 64)
    gt = 2 * pred + 1
    loss, abs_rel, count = masked_ssi_l1(pred, gt, mask)
    assert loss.item() < 1e-5, f"affine-aligned loss should be ~0, got {loss.item()}"
    assert count == 1600, count

    # Noisy GT: positive loss and non-zero gradient through the prediction.
    pred2 = torch.rand(1, 64, 64, requires_grad=True)
    gt2 = 2 * pred2.detach() + 1 + 0.1 * torch.randn(1, 64, 64)
    loss2, _, _ = masked_ssi_l1(pred2, gt2, mask)
    loss2.backward()
    assert loss2.item() > 0
    assert pred2.grad is not None and float(pred2.grad.norm()) > 0

    # Constant prediction: lstsq still finite (degenerate scale), loss finite.
    pred3 = torch.full((1, 64, 64), 0.5, requires_grad=True)
    gt3 = torch.rand(1, 64, 64)
    loss3, _, _ = masked_ssi_l1(pred3, gt3, mask)
    assert bool(torch.isfinite(loss3))

    # Empty mask must be rejected.
    try:
        masked_ssi_l1(pred, gt, torch.zeros(1, 64, 64))
    except RuntimeError:
        pass
    else:
        raise AssertionError("empty mask was not rejected")

    print("SSI loss checks passed: affine-zero, gradient flow, degenerate, empty-mask guard.")


if __name__ == "__main__":
    main()
