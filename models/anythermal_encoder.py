"""Standalone AnyThermal/DINOv2 feature extraction wrapper.

This module intentionally does not depend on Lotus, Marigold, or diffusion code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, TypedDict, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ImageLike = Union[str, Path, Image.Image, np.ndarray, torch.Tensor]


class ThermalInputDiagnostics(TypedDict):
    loading_mode: str
    pil_mode: str
    original_numpy_dtype: str
    original_array_shape: Tuple[int, ...]
    raw_min: float
    raw_max: float
    raw_mean: float
    raw_std: float
    converted_uint8_min: float
    converted_uint8_max: float
    converted_uint8_mean: float
    converted_uint8_std: float
    rgb_channels_equal: bool
    processor_pixel_values_shape: Tuple[int, ...]
    processor_pixel_values_mean: float
    processor_pixel_values_std: float
    legacy_rgb_min: float
    legacy_rgb_max: float
    legacy_rgb_mean: float
    legacy_rgb_std: float
    legacy_rgb_saturated_fraction: float

class AnyThermalEncoderOutput(TypedDict):
    last_hidden_state: torch.Tensor
    cls_token: torch.Tensor
    register_tokens: torch.Tensor
    patch_tokens: torch.Tensor
    spatial_features: torch.Tensor
    grid_size: Tuple[int, int]
    num_register_tokens: int
    original_shape: Tuple[int, ...]
    preprocessed_shape: Tuple[int, ...]
    model_device: torch.device
    thermal_diagnostics: ThermalInputDiagnostics

class ParameterSummary(TypedDict):
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    is_fully_frozen: bool

VIT_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
VIT_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


class AnyThermalEncoder:
    """Load an AnyThermal/HF DINOv2 encoder and return transformer tokens."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        *,
        processor_path: Optional[str] = None,
        revision: Optional[str] = None,
        trust_remote_code: bool = True,
        dtype: Optional[torch.dtype] = None,
        local_files_only: bool = False,
        fallback_preprocess: bool = True,
    ) -> None:
        self.model_path = model_path
        self.processor_path = processor_path or model_path
        self.revision = revision
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self._last_raw_thermal_array: Optional[np.ndarray] = None
        self._last_processor_image: Optional[Image.Image] = None
        self._last_thermal_diagnostics: Optional[ThermalInputDiagnostics] = None

        self.processor = self._load_processor(fallback_preprocess=fallback_preprocess)
        self.model = self._load_model().to(self.device)
        self.num_register_tokens = self._detect_num_register_tokens()
        self.patch_size = self._detect_patch_size(default=14)

        if dtype is not None:
            self.model = self.model.to(dtype=dtype)

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def parameter_summary(self) -> ParameterSummary:
        """Return parameter counts and the encoder's frozen state."""
        total_parameters = sum(parameter.numel() for parameter in self.model.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        frozen_parameters = total_parameters - trainable_parameters
        return {
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
            "frozen_parameters": frozen_parameters,
            "is_fully_frozen": trainable_parameters == 0,
        }

    def encode(self, thermal_image: ImageLike) -> AnyThermalEncoderOutput:
        """Run one forward pass on a thermal image.

        Returns:
            Token tensors plus preprocessing and model metadata. Register tokens
            are returned separately and removed from patch tokens.
        """
        inputs, original_shape = self.preprocess(thermal_image)

        with torch.inference_mode():
            outputs = self.model(**inputs)

        if not hasattr(outputs, "last_hidden_state"):
            raise RuntimeError("Model output does not contain last_hidden_state.")

        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0]
        register_start = 1
        patch_start = register_start + self.num_register_tokens
        register_tokens = last_hidden_state[:, register_start:patch_start]
        patch_tokens = last_hidden_state[:, patch_start:]

        preprocessed_shape = tuple(inputs["pixel_values"].shape)
        if len(preprocessed_shape) != 4:
            raise RuntimeError(
                "Expected preprocessed pixel_values shape [B,C,H,W], "
                f"got {preprocessed_shape}."
            )
        if patch_tokens.ndim != 3:
            raise RuntimeError(
                "Expected patch tokens shape [B,N,C], "
                f"got {tuple(patch_tokens.shape)}."
            )

        preprocessed_height, preprocessed_width = preprocessed_shape[-2:]
        grid_height = preprocessed_height // self.patch_size
        grid_width = preprocessed_width // self.patch_size
        expected_patch_tokens = grid_height * grid_width
        actual_patch_tokens = patch_tokens.shape[1]
        if actual_patch_tokens != expected_patch_tokens:
            raise RuntimeError(
                "Cannot restore AnyThermal spatial features: patch token count "
                f"{actual_patch_tokens} does not match grid {grid_height}x{grid_width}="
                f"{expected_patch_tokens} for preprocessed shape {preprocessed_shape} "
                f"and patch size {self.patch_size}."
            )

        batch_size, _, channels = patch_tokens.shape
        spatial_features = (
            patch_tokens.reshape(batch_size, grid_height, grid_width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return {
            "last_hidden_state": last_hidden_state,
            "cls_token": cls_token,
            "register_tokens": register_tokens,
            "patch_tokens": patch_tokens,
            "spatial_features": spatial_features,
            "grid_size": (grid_height, grid_width),
            "num_register_tokens": self.num_register_tokens,
            "original_shape": original_shape,
            "preprocessed_shape": preprocessed_shape,
            "model_device": next(self.model.parameters()).device,
            "thermal_diagnostics": self._require_thermal_diagnostics(),
        }

    def preprocess(self, thermal_image: ImageLike) -> Tuple[Dict[str, torch.Tensor], Tuple[int, ...]]:
        """Convert a thermal image to model inputs without losing high-bit-depth data."""
        image, original_shape, diagnostics, raw_array = self._to_rgb_pil(thermal_image)
        self._last_raw_thermal_array = raw_array.copy()
        self._last_processor_image = image.copy()

        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = self._move_inputs(inputs)
        else:
            tensor = self._manual_anythermal_preprocess(image)
            inputs = {"pixel_values": tensor.to(self.device)}

        pixel_values = inputs["pixel_values"]
        diagnostics["processor_pixel_values_shape"] = tuple(pixel_values.shape)
        diagnostics["processor_pixel_values_mean"] = float(pixel_values.float().mean())
        diagnostics["processor_pixel_values_std"] = float(pixel_values.float().std())
        self._last_thermal_diagnostics = diagnostics
        return inputs, original_shape

    def get_last_thermal_artifacts(self) -> Tuple[np.ndarray, Image.Image]:
        """Return copies of the raw thermal array and RGB processor input."""
        if self._last_raw_thermal_array is None or self._last_processor_image is None:
            raise RuntimeError("No thermal image has been preprocessed yet.")
        return self._last_raw_thermal_array.copy(), self._last_processor_image.copy()

    def _require_thermal_diagnostics(self) -> ThermalInputDiagnostics:
        if self._last_thermal_diagnostics is None:
            raise RuntimeError("No thermal diagnostics are available.")
        return self._last_thermal_diagnostics.copy()
    def _load_processor(self, *, fallback_preprocess: bool) -> Optional[Any]:
        try:
            from transformers import AutoImageProcessor

            return AutoImageProcessor.from_pretrained(
                self.processor_path,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            if fallback_preprocess:
                self.processor_load_error = exc
                return None
            raise RuntimeError(
                f"Failed to load image processor from '{self.processor_path}'."
            ) from exc

    def _load_model(self) -> torch.nn.Module:
        try:
            from transformers import AutoModel

            return AutoModel.from_pretrained(
                self.model_path,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load model from '{self.model_path}'.") from exc

    def _move_inputs(self, inputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        moved = {}
        for name, value in inputs.items():
            if torch.is_tensor(value):
                value = value.to(self.device)
                if name == "pixel_values" and self.dtype is not None:
                    value = value.to(dtype=self.dtype)
            moved[name] = value
        return moved

    def _manual_anythermal_preprocess(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        _, _, height, width = tensor.shape
        new_height = max(self.patch_size, (height // self.patch_size) * self.patch_size)
        new_width = max(self.patch_size, (width // self.patch_size) * self.patch_size)
        tensor = F.interpolate(
            tensor,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        mean = torch.tensor(VIT_IMAGE_MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor(VIT_IMAGE_STD, dtype=tensor.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        if self.dtype is not None:
            tensor = tensor.to(dtype=self.dtype)
        return tensor

    @staticmethod
    def _to_rgb_pil(
        thermal_image: ImageLike,
    ) -> Tuple[Image.Image, Tuple[int, ...], ThermalInputDiagnostics, np.ndarray]:
        legacy_rgb: Optional[np.ndarray] = None

        if isinstance(thermal_image, (str, Path)):
            with Image.open(thermal_image) as source_image:
                pil_mode = source_image.mode
                raw_array = np.asarray(source_image).copy()
                legacy_rgb = np.asarray(source_image.convert("RGB")).copy()
            loading_mode = "path"
            original_shape = AnyThermalEncoder._image_metadata_shape(raw_array)
        elif isinstance(thermal_image, Image.Image):
            pil_mode = thermal_image.mode
            raw_array = np.asarray(thermal_image).copy()
            legacy_rgb = np.asarray(thermal_image.convert("RGB")).copy()
            loading_mode = "PIL.Image"
            original_shape = AnyThermalEncoder._image_metadata_shape(raw_array)
        elif isinstance(thermal_image, torch.Tensor):
            tensor = thermal_image.detach().cpu()
            original_shape = tuple(tensor.shape)
            if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            if tensor.ndim == 2:
                raw_array = tensor.numpy()
            elif tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
                raw_array = tensor.numpy()
            else:
                raise ValueError(
                    "Expected thermal tensor shape [H,W], [1,H,W], [3,H,W], "
                    "[H,W,1], or [H,W,3]."
                )
            pil_mode = "not-applicable"
            loading_mode = "torch.Tensor"
        elif isinstance(thermal_image, np.ndarray):
            original_shape = tuple(thermal_image.shape)
            raw_array = np.asarray(thermal_image)
            pil_mode = "not-applicable"
            loading_mode = "numpy.ndarray"
        else:
            raise TypeError(f"Unsupported thermal image type: {type(thermal_image)!r}")

        if raw_array.size == 0:
            raise ValueError("Thermal image is empty.")
        raw_values = np.asarray(raw_array, dtype=np.float64)
        if not np.isfinite(raw_values).all():
            raise ValueError("Thermal image contains NaN or infinite values.")

        uint8_array = AnyThermalEncoder._array_to_uint8(raw_array)
        if uint8_array.ndim == 2:
            grayscale = Image.fromarray(uint8_array, mode="L")
        elif uint8_array.ndim == 3 and uint8_array.shape[-1] == 1:
            grayscale = Image.fromarray(uint8_array[..., 0], mode="L")
        elif uint8_array.ndim == 3 and uint8_array.shape[-1] == 3:
            if not (
                np.array_equal(uint8_array[..., 0], uint8_array[..., 1])
                and np.array_equal(uint8_array[..., 1], uint8_array[..., 2])
            ):
                raise ValueError(
                    "Three-channel thermal input must have identical channels; "
                    "pseudocolor or RGB content is not accepted."
                )
            grayscale = Image.fromarray(uint8_array[..., 0], mode="L")
        else:
            raise ValueError(
                "Expected thermal image shape [H,W], [H,W,1], or [H,W,3], "
                f"got {tuple(uint8_array.shape)}."
            )

        rgb_image = grayscale.convert("RGB")
        rgb_array = np.asarray(rgb_image)
        channels_equal = bool(
            np.array_equal(rgb_array[..., 0], rgb_array[..., 1])
            and np.array_equal(rgb_array[..., 1], rgb_array[..., 2])
        )
        if not channels_equal:
            raise RuntimeError("Thermal RGB channel replication failed.")

        converted_values = uint8_array.astype(np.float64)
        if converted_values.ndim == 3:
            converted_values = converted_values[..., 0]
        if legacy_rgb is None:
            legacy_values = np.zeros(1, dtype=np.float64)
            legacy_saturated_fraction = 0.0
        else:
            legacy_values = legacy_rgb.astype(np.float64)
            legacy_saturated_fraction = float((legacy_rgb == 255).mean())

        diagnostics: ThermalInputDiagnostics = {
            "loading_mode": loading_mode,
            "pil_mode": pil_mode,
            "original_numpy_dtype": str(raw_array.dtype),
            "original_array_shape": tuple(raw_array.shape),
            "raw_min": float(raw_values.min()),
            "raw_max": float(raw_values.max()),
            "raw_mean": float(raw_values.mean()),
            "raw_std": float(raw_values.std()),
            "converted_uint8_min": float(converted_values.min()),
            "converted_uint8_max": float(converted_values.max()),
            "converted_uint8_mean": float(converted_values.mean()),
            "converted_uint8_std": float(converted_values.std()),
            "rgb_channels_equal": channels_equal,
            "processor_pixel_values_shape": (),
            "processor_pixel_values_mean": 0.0,
            "processor_pixel_values_std": 0.0,
            "legacy_rgb_min": float(legacy_values.min()),
            "legacy_rgb_max": float(legacy_values.max()),
            "legacy_rgb_mean": float(legacy_values.mean()),
            "legacy_rgb_std": float(legacy_values.std()),
            "legacy_rgb_saturated_fraction": legacy_saturated_fraction,
        }
        return rgb_image, original_shape, diagnostics, raw_array

    @staticmethod
    def _image_metadata_shape(array: np.ndarray) -> Tuple[int, ...]:
        if array.ndim == 2:
            return (array.shape[0], array.shape[1], 1)
        return tuple(array.shape)
    @staticmethod
    def _array_to_uint8(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.dtype == np.uint8:
            return array

        array = array.astype(np.float32)
        finite = np.isfinite(array)
        if not finite.all():
            raise ValueError("Thermal image contains NaN or infinite values.")

        min_value = float(array.min())
        max_value = float(array.max())
        if min_value >= 0.0 and max_value <= 1.0:
            array = array * 255.0
        elif min_value < 0.0 or max_value > 255.0:
            if max_value == min_value:
                array = np.zeros_like(array)
            else:
                array = (array - min_value) / (max_value - min_value) * 255.0

        return np.clip(array, 0, 255).round().astype(np.uint8)

    def _detect_num_register_tokens(self) -> int:
        candidates = [
            self.model,
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "config", None), "vision_config", None),
            getattr(getattr(self.model, "config", None), "backbone_config", None),
        ]
        for obj in candidates:
            value = getattr(obj, "num_register_tokens", None)
            if value is not None:
                return int(value)
        return 0

    def _detect_patch_size(self, *, default: int) -> int:
        candidates = [
            self.model,
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "config", None), "vision_config", None),
            getattr(getattr(self.model, "config", None), "backbone_config", None),
        ]
        for obj in candidates:
            value = getattr(obj, "patch_size", None)
            if value is not None:
                if isinstance(value, (tuple, list)):
                    return int(value[0])
                return int(value)
        return default

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device}', but CUDA is not available. Use --device cpu or run in a CUDA environment."
            )
        return torch.device(device)

