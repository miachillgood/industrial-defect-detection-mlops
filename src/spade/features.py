"""Frozen ImageNet backbone used as SPADE's feature pyramid.

SPADE does not train anything: it extracts multi-scale activations from a frozen
ImageNet classifier and does nearest-neighbour retrieval against the features of
the (defect-free) training images.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


class PyramidFeatureExtractor(nn.Module):
    """Wide-ResNet50-2 with forward hooks on layer1/2/3 and the global avgpool.

    ``layer1..3`` give the deep pyramid used for pixel-level correspondence;
    ``avgpool`` gives the 2048-d global descriptor used for image-level kNN.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if backbone != "wide_resnet50_2":
            raise ValueError(
                f"backbone {backbone!r} is not supported; this pipeline targets "
                "wide_resnet50_2 (IMAGENET1K_V1), as in the reference implementation"
            )
        # IMAGENET1K_V1 is exactly what the legacy ``pretrained=True`` resolved to.
        # The newer V2 weights would silently change every number in the report.
        self.net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.net.eval()
        self.net.to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)

        self.layers = tuple(layers)
        self.device = device
        self._buffer: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._handles = []
        self._register_hooks()

    # ------------------------------------------------------------------ hooks
    def _register_hooks(self) -> None:
        def make_hook(name: str):
            def hook(_module, _inp, out):
                self._buffer[name] = out

            return hook

        for name in self.layers:
            block = getattr(self.net, name)
            # Hook the last bottleneck of the stage, matching the reference.
            self._handles.append(block[-1].register_forward_hook(make_hook(name)))
        self._handles.append(self.net.avgpool.register_forward_hook(make_hook("avgpool")))

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        self._buffer.clear()
        self.net(x.to(self.device, non_blocking=True))
        return OrderedDict((k, v) for k, v in self._buffer.items())

    @property
    def feature_names(self) -> tuple[str, ...]:
        return (*self.layers, "avgpool")


@torch.no_grad()
def extract_features(
    extractor: PyramidFeatureExtractor,
    dataloader,
    dtype: torch.dtype = torch.float32,
    progress_desc: str | None = None,
    collect_inputs: bool = False,
):
    """Run the extractor over a dataloader.

    Returns ``(features, labels, masks, images)`` where ``features`` maps each
    hooked layer name to a stacked CPU tensor. ``images`` is only populated when
    ``collect_inputs`` is set (needed for the localisation figures).
    """
    from tqdm import tqdm

    chunks: dict[str, list[torch.Tensor]] = {k: [] for k in extractor.feature_names}
    labels: list[int] = []
    masks: list[torch.Tensor] = []
    images: list[torch.Tensor] = []

    iterator = dataloader
    if progress_desc:
        iterator = tqdm(dataloader, desc=progress_desc, leave=False)

    for x, y, mask in iterator:
        out = extractor(x)
        for name, value in out.items():
            chunks[name].append(value.detach().to("cpu", dtype=dtype))
        labels.extend(int(v) for v in y)
        masks.append(mask.to(torch.uint8))
        if collect_inputs:
            images.append(x.clone())

    features = {k: torch.cat(v, dim=0) for k, v in chunks.items()}
    mask_tensor = torch.cat(masks, dim=0) if masks else torch.empty(0)
    image_tensor = torch.cat(images, dim=0) if images else torch.empty(0)
    return features, torch.tensor(labels, dtype=torch.long), mask_tensor, image_tensor
