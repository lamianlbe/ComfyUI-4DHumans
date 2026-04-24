"""
Runtime device-state probe for diagnosing CPU-vs-GPU surprises.

Ultralytics wrappers have a THREE-LAYER device state that can get
out of sync:

  1. ``yolo.device`` — cached wrapper attribute, sometimes stale
  2. ``yolo.model.device`` — set at construction, sometimes stale too
  3. ``next(yolo.model.parameters()).device`` — the GROUND TRUTH of
     where the weights actually live

Pair that with CUDA memory state and you can tell at a glance whether
a ``.to("cuda")`` call actually did anything or silently failed.
"""

import logging
import torch

_logger = logging.getLogger(__name__)


def probe(obj, tag: str) -> str:
    """Return a single-line summary of ``obj``'s device + dtype state.

    Accepts any of:
      * ``nn.Module`` directly
      * Ultralytics ``YOLO`` wrapper (has ``.model`` inner nn.Module)
      * Ultralytics predictor (has ``.model`` attribute)
    """
    parts = []

    # --- CUDA state -----------------------------------------------------
    try:
        cuda_ok = torch.cuda.is_available()
        parts.append(f"cuda={cuda_ok}")
        if cuda_ok:
            alloc = torch.cuda.memory_allocated() / (1 << 30)
            reserv = torch.cuda.memory_reserved() / (1 << 30)
            parts.append(f"vram={alloc:.2f}/{reserv:.2f}GiB")
    except Exception as e:
        parts.append(f"cuda-probe-err={e}")

    # --- Wrapper .device attribute (Ultralytics caches this) -----------
    wd = getattr(obj, "device", None)
    if wd is not None:
        parts.append(f"wrap.device={wd}")

    # --- Inner nn.Module weights (the ground truth) --------------------
    # Ultralytics wrappers expose the actual nn.Module at `.model`; plain
    # nn.Module doesn't have a `.model` attribute but `parameters()`
    # works directly.
    inner = getattr(obj, "model", None)
    source = inner if (inner is not None and hasattr(inner, "parameters")) else obj
    if hasattr(source, "parameters"):
        try:
            p = next(source.parameters())
            parts.append(f"weights={p.device}({str(p.dtype).replace('torch.', '')})")
        except StopIteration:
            parts.append("weights=no-params")
        except Exception as e:
            parts.append(f"weights-err={e}")

    return f"[probe {tag}] " + " | ".join(parts)


def log_probe(obj, tag: str, level: int = logging.INFO):
    """Emit ``probe(obj, tag)`` to the module logger at INFO by default."""
    _logger.log(level, probe(obj, tag))
