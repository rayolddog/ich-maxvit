"""
HU Window Skill — Hounsfield Unit mapping for CT tensor caches.

Provides:
  - Three calibrated HUWindow presets (wide, medium, narrow)
  - apply_window(): HU float32 array → float16 tensor in [0.0, 1.0]

Design principles:
  - All mappings use FIXED global HU bounds — no per-image normalisation.
    Per-image scaling destroys the absolute HU calibration that distinguishes
    tissue types (e.g. acute ICH at 50–80 HU vs. gray matter at 30–42 HU).
  - Conversion from raw DICOM pixels to HU always uses DICOM header metadata
    (RescaleSlope, RescaleIntercept) — never inferred from pixel statistics.
  - This module is independent of the rest of the pipeline; import it anywhere.

HU reference values (non-contrast CT head):
  Air (external)        ≈ −1000
  Fat                   ≈ −100 to −50
  Water / CSF           ≈    0 to  15
  White matter          ≈   20 to  30
  Gray matter           ≈   30 to  42
  Acute ICH / blood     ≈   50 to  80
  Hyperacute clot       ≈   80 to 100
  Cortical bone         ≈  300 to 1000
  Metal / hardware      >  1000
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class HUWindow:
    """Defines a linear mapping from a HU range to [0.0, 1.0].

    Values below hu_low  map to 0.0.
    Values above hu_high map to 1.0.
    Values in between are mapped linearly.
    """
    name:    str
    hu_low:  float   # HU → 0.0
    hu_high: float   # HU → 1.0

    @property
    def hu_range(self) -> float:
        return self.hu_high - self.hu_low

    def __str__(self) -> str:
        return (f"HUWindow({self.name}: "
                f"{self.hu_low:.0f} to {self.hu_high:.0f} HU, "
                f"range={self.hu_range:.0f})")


# ── Three calibrated presets ────────────────────────────────────────────────

WINDOW_WIDE = HUWindow(
    name     = 'wide',
    hu_low   = -1024.0,
    hu_high  =  3071.0,
)
"""
Wide window: full practical CT range.

  hu_low  = −1024  (CT table air, scanner minimum)
  hu_high =  3071  (dense bone / metal upper practical limit)
  range   =  4095  (≈ 12-bit dynamic range)

Maps virtually all CT values without saturation. Bone and metal are
clearly separated from soft tissue, but the contrast between similar soft
tissues (e.g. ICH vs. gray matter) is compressed to a very small fraction
of the [0, 1] range.

  gray matter (37 HU) → (37 + 1024) / 4095 ≈ 0.26
  acute ICH   (65 HU) → (65 + 1024) / 4095 ≈ 0.27
  delta ≈ 0.007

Use for: multi-tissue classification, bone/implant tasks, scout-level survey.
"""

WINDOW_MEDIUM = HUWindow(
    name     = 'medium',
    hu_low   = -200.0,
    hu_high  =  200.0,
)
"""
Medium window: soft-tissue and early-bone range.

  hu_low  = −200  (excludes bulk air; includes sinuses, fat edges)
  hu_high =  200  (lower cortical bone / dense calcification)
  range   =   400

Captures fat, water, CSF, brain parenchyma, blood, and early bone.
Air and dense bone are saturated to 0.0 / 1.0 respectively.

  gray matter (37 HU) → (37 + 200) / 400 = 0.593
  acute ICH   (65 HU) → (65 + 200) / 400 = 0.663
  delta ≈ 0.070   (10× wider than wide window)

Use for: brain anatomy, ICH in anatomical context, soft-tissue masses.
"""

WINDOW_NARROW = HUWindow(
    name     = 'narrow',
    hu_low   =  48.0,
    hu_high  =  90.0,
)
"""
Narrow window: acute ICH detection range.

  hu_low  =  48  (lower edge of acute hemorrhage; above gray matter ceiling)
  hu_high =  90  (upper edge of hyperacute clot)
  range   =  42

Everything below 48 HU (CSF, gray matter, white matter, fat, air) maps to
0.0.  Everything above 90 HU (bone, metal) maps to 1.0.  Only the acute
blood density band occupies the full [0, 1] output range.

  gray matter (37 HU) → 0.0   (below window)
  CSF         (10 HU) → 0.0   (below window)
  acute ICH   (65 HU) → (65 − 48) / 42 = 0.405
  dense clot  (85 HU) → (85 − 48) / 42 = 0.881

Use for: maximum ICH vs. normal brain pixel contrast; binary ICH screening.
"""

WINDOW_BALANCED = HUWindow(
    name    = 'balanced',
    hu_low  = -512.0,
    hu_high =  512.0,
)
"""
Balanced window: broad soft-tissue range with bone and air as anchors.

  hu_low  = −512  (far below brain; anchors air/fat end)
  hu_high =  512  (lower dense bone; anchors bony end)
  range   = 1024

Captures the full soft-tissue spectrum (fat → CSF → brain → ICH → early bone)
with ~4× better ICH/gray contrast than WINDOW_WIDE while keeping anatomical
context (skull outline, sinuses, scalp) within the linear range.

  air (external) (−1000 HU) → 0.0  (clipped)
  fat             (−75 HU)  → (−75 + 512) / 1024 ≈ 0.427
  CSF / water      (10 HU)  → (10  + 512) / 1024 ≈ 0.510
  gray matter      (37 HU)  → (37  + 512) / 1024 ≈ 0.536
  acute ICH        (65 HU)  → (65  + 512) / 1024 ≈ 0.564
  dense bone      (512 HU)  → 1.0  (at window edge)
  ICH vs gray delta ≈ 0.028  (4× wider than WINDOW_WIDE's 0.007)

Use for: CLAHE-enhanced 1-channel ICH detection with anatomical context.
Pair with CLAHE (n_bins=256; 1024 HU / 256 bins ≈ 4 HU/bin resolution) to
amplify local tissue contrast without sacrificing the calibrated HU mapping.
"""


# ── Core mapping function ────────────────────────────────────────────────────

def apply_window(hu: np.ndarray, window: HUWindow) -> np.ndarray:
    """Map a float32 HU array to float16 in [0.0, 1.0] using a fixed window.

    Args:
        hu:     float32 ndarray of any shape, values in Hounsfield Units.
        window: one of WINDOW_WIDE, WINDOW_MEDIUM, WINDOW_NARROW (or custom).

    Returns:
        float16 ndarray, same shape as hu, values clipped to [0.0, 1.0].

    No per-image scaling is performed.  The mapping is deterministic and
    reproducible: the same HU value always produces the same output regardless
    of which image it came from.
    """
    normed = (hu.astype(np.float32) - window.hu_low) / window.hu_range
    normed = np.clip(normed, 0.0, 1.0)
    return normed.astype(np.float16)
