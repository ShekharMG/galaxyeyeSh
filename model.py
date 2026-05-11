"""
model.py — Siamese UNet for SAR-EO Change Detection

Architecture overview:
  ┌─────────────┐     ┌─────────────┐
  │  Pre (RGB)  │     │ Post (SAR)  │
  │   3-ch      │     │   1-ch      │
  └──────┬──────┘     └──────┬──────┘
         │  separate          │  separate
         │  stem conv         │  stem conv
         ▼                    ▼
  ┌──────────────────────────────────┐
  │       Shared Encoder             │
  │  (ResNet34 / EfficientNet-B2)    │
  │  — weight-shared feature         │
  │    extraction at 5 scales        │
  └──────────────────────────────────┘
         │                    │
    feat_pre [s1…s5]    feat_post [s1…s5]
         │                    │
         └───────┬────────────┘
                 │  FUSION (per scale):
                 │  concat(pre, post, |pre - post|)
                 ▼
  ┌──────────────────────────────────┐
  │          UNet Decoder            │
  │  (skip connections + upsampling) │
  └──────────────┬───────────────────┘
                 │
                 ▼
          sigmoid output (H×W)

Key design decisions:
  - We use segmentation-models-pytorch (smp) to get a production-quality
    encoder + decoder, but we CANNOT use smp.Unet directly because it
    expects a single input.  Instead we extract the encoder from smp and
    build a custom Siamese wrapper.
  - Separate first-conv ("stem") layers handle the channel mismatch:
    pre-event has 3 channels (RGB), post-event has 1 channel (SAR).
    After the stem, both branches produce the same number of feature maps,
    allowing weight sharing in the deeper encoder layers.
  - Feature fusion uses concatenation + absolute difference at each skip
    level, giving the decoder both correspondence and discrepancy cues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SiameseChangeUNet(nn.Module):
    """
    Siamese UNet for binary change detection.

    Args:
        encoder_name:  backbone name (e.g. 'resnet34', 'efficientnet-b2')
        pretrained:    use ImageNet pretrained weights
        in_channels_pre:  number of input channels for pre-event (default=3, RGB)
        in_channels_post: number of input channels for post-event (default=1, SAR)
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        pretrained: bool = True,
        in_channels_pre: int = 3,
        in_channels_post: int = 1,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        weights = "imagenet" if pretrained else None

        # ---------------------------------------------------------------
        # 1. Create a reference smp.Unet to extract encoder architecture
        #    and get the decoder for free.
        # ---------------------------------------------------------------
        self._ref_unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=3,  # Reference uses 3ch
            classes=1,
        )

        # The shared encoder (we'll modify the first conv)
        self.encoder = self._ref_unet.encoder

        # ---------------------------------------------------------------
        # 2. Separate stem convolutions for pre (3ch) and post (1ch)
        #    This handles the input channel mismatch while allowing
        #    the rest of the encoder to share weights.
        # ---------------------------------------------------------------
        # Get the original first conv layer info
        original_first_conv = self._get_first_conv(self.encoder)
        out_channels = original_first_conv.out_channels
        kernel_size = original_first_conv.kernel_size
        stride = original_first_conv.stride
        padding = original_first_conv.padding

        # Stem for pre-event (3-channel RGB) — initialized from pretrained weights
        self.stem_pre = nn.Conv2d(
            in_channels_pre, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        # Copy pretrained weights for the 3-channel stem
        with torch.no_grad():
            self.stem_pre.weight.copy_(original_first_conv.weight)

        # Stem for post-event (1-channel SAR) — initialized by averaging RGB weights
        self.stem_post = nn.Conv2d(
            in_channels_post, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        with torch.no_grad():
            # Average across the 3 input-channel dimension → (out_ch, 1, kH, kW)
            self.stem_post.weight.copy_(
                original_first_conv.weight.mean(dim=1, keepdim=True)
            )

        # ---------------------------------------------------------------
        # 3. Build a new decoder that accepts fused (concat+diff) features
        #    The fused features have 3× the channel count at each skip level:
        #    [pre_feat, post_feat, |pre_feat - post_feat|]
        # ---------------------------------------------------------------
        encoder_channels = self.encoder.out_channels  # e.g. (3, 64, 64, 128, 256, 512)

        # Fused channels: 3x at each level (pre + post + |diff|)
        # But the first level (input-resolution) we handle via stems
        fused_channels = list(ch * 3 for ch in encoder_channels)
        # At i=0, feats_pre is 3-ch, feats_post is 1-ch. Diff broadcasts to 3-ch. Total = 3 + 1 + 3 = 7.
        fused_channels[0] = in_channels_pre + in_channels_post + max(in_channels_pre, in_channels_post)

        # 1x1 conv adapters to reduce fused channels back to encoder channels
        # so we can reuse smp's standard UnetDecoder
        self.fusion_adapters = nn.ModuleList()
        for i, (fused_ch, enc_ch) in enumerate(zip(fused_channels, encoder_channels)):
            if enc_ch == 0 or fused_ch == 0:
                self.fusion_adapters.append(nn.Identity())
            else:
                self.fusion_adapters.append(
                    nn.Sequential(
                        nn.Conv2d(fused_ch, enc_ch, kernel_size=1, bias=False),
                        nn.BatchNorm2d(enc_ch),
                        nn.ReLU(inplace=True),
                    )
                )

        # Reuse the smp decoder
        self.decoder = self._ref_unet.decoder
        self.segmentation_head = self._ref_unet.segmentation_head

    def _get_first_conv(self, encoder):
        """Extract the first convolutional layer from various encoder architectures."""
        # ResNet family
        if hasattr(encoder, "conv1"):
            return encoder.conv1
        # EfficientNet family (via timm)
        if hasattr(encoder, "_conv_stem"):
            return encoder._conv_stem
        # Generic fallback: walk children
        for module in encoder.modules():
            if isinstance(module, nn.Conv2d):
                return module
        raise RuntimeError("Could not find the first Conv2d in the encoder.")

    def _set_first_conv(self, encoder, new_conv):
        """Replace the first conv layer in the encoder with a given conv."""
        if hasattr(encoder, "conv1"):
            encoder.conv1 = new_conv
        elif hasattr(encoder, "_conv_stem"):
            encoder._conv_stem = new_conv
        else:
            raise RuntimeError("Could not replace the first Conv2d.")

    def _encode(self, x, stem):
        """
        Run an input through the given stem, then through the shared encoder.
        We temporarily swap the encoder's first conv with our stem.
        """
        # Save original first conv
        original_conv = self._get_first_conv(self.encoder)
        # Swap in our stem
        self._set_first_conv(self.encoder, stem)
        # Forward pass through encoder
        features = self.encoder(x)
        # Restore original conv
        self._set_first_conv(self.encoder, original_conv)
        return features

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pre:  (B, 3, H, W) pre-event EO image
            post: (B, 1, H, W) post-event SAR image

        Returns:
            (B, 1, H, W) sigmoid-activated change probability map
        """
        # Extract multi-scale features from both branches
        feats_pre = self._encode(pre, self.stem_pre)    # list of tensors
        feats_post = self._encode(post, self.stem_post)  # list of tensors

        # Fuse features at each scale: concat(pre, post, |pre - post|)
        fused_feats = []
        for i, (fp, fq) in enumerate(zip(feats_pre, feats_post)):
            diff = torch.abs(fp - fq)
            fused = torch.cat([fp, fq, diff], dim=1)  # 3× channels
            fused = self.fusion_adapters[i](fused)     # reduce back to enc channels
            fused_feats.append(fused)

        # Decode fused features
        decoder_output = self.decoder(*fused_feats)

        # Segmentation head produces logits → sigmoid
        logits = self.segmentation_head(decoder_output)
        return logits


def build_model(cfg: dict) -> SiameseChangeUNet:
    """
    Factory function to create the model from config.

    Args:
        cfg: parsed config dict with keys 'encoder', 'pretrained'

    Returns:
        SiameseChangeUNet instance
    """
    encoder_name = cfg.get("encoder", "resnet34")
    pretrained = cfg.get("pretrained", True)

    model = SiameseChangeUNet(
        encoder_name=encoder_name,
        pretrained=pretrained,
        in_channels_pre=3,
        in_channels_post=1,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {encoder_name} Siamese UNet")
    print(f"[Model] Total params: {total_params:,}")
    print(f"[Model] Trainable params: {trainable_params:,}")

    return model
