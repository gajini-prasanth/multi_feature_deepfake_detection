from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def overlay_heatmap(rgb_image: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a [0, 1] 2D heatmap onto a uint8 RGB image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (1.0 - alpha) * rgb_image.astype(np.float32) + alpha * heatmap.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


class MultiStreamGradCAM:
    """Generate Grad-CAM heatmaps for each enabled stream in MultiFeatureFusionModel."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self._handles = []

        for name, stream in self.model.stream_modules.items():
            # Hook the final spatial layer of each stream encoder
            target_layer = stream.encoder[-1]
            self._handles.append(target_layer.register_forward_hook(self._make_fwd_hook(name)))
            self._handles.append(target_layer.register_full_backward_hook(self._make_bwd_hook(name)))

    def _make_fwd_hook(self, name: str):
        def hook(_module, _inp, out):
            self.activations[name] = out
        return hook

    def _make_bwd_hook(self, name: str):
        def hook(_module, _grad_in, grad_out):
            self.gradients[name] = grad_out[0]
        return hook

    def generate(self, x: torch.Tensor) -> Dict[str, object]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        logit = out["logits"][0]
        logit.backward()

        h, w = x.shape[-2:]
        cams: Dict[str, np.ndarray] = {}
        for name in self.model.stream_modules:
            act = self.activations[name][0]  # [C, H', W']
            grad = self.gradients[name][0]  # [C, H', W']
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = F.relu((weights * act).sum(dim=0, keepdim=True)).unsqueeze(0)
            cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            cam_np = cam.detach().cpu().numpy()
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
            cams[name] = cam_np

        combined = np.mean(list(cams.values()), axis=0)
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)
        prob = float(torch.sigmoid(logit).item())
        return {"probability": prob, "cams": cams, "combined_cam": combined}

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
