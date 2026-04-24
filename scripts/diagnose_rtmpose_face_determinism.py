"""
Diagnose whether RTMPose-Face ONNX output drift comes from:
  (A) session-state pollution between consecutive session.run() calls, OR
  (B) genuine model sensitivity to tiny input differences (bimodal heatmaps)

Usage (on the machine that has the ONNX model installed):

    python diagnose_rtmpose_face_determinism.py \\
        /path/to/rtmface_106_debug.npz \\
        /path/to/models/rtmpose-face/rtmpose-m-face.onnx

It runs 3 scenarios:

  1) "IDENTICAL x8"  — feed the SAME image (f0's preproc) 8 times in a row.
                       If outputs alternate here, the session is stateful
                       and the bug is A. If they're stable, the bug is B.

  2) "REAL SEQ x4"   — feed the actual 4 frames from the npz and confirm
                       we reproduce the alternating pattern.

  3) "FRESH SESSION" — create a fresh InferenceSession for each of the 4
                       frames. If this is stable but REAL SEQ isn't,
                       session state reuse is the smoking gun.

Compares landmark 0 and 32 (jaw endpoints — known to flip) across runs.
"""

import sys

import numpy as np
import onnxruntime as ort


def make_session(onnx_path, provider="cpu"):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.enable_mem_pattern = False
    so.enable_cpu_mem_arena = False
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if provider == "cuda":
        providers = [
            ("CUDAExecutionProvider", {
                "cudnn_conv_algo_search": "DEFAULT",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


_IMG_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMG_STD  = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def preproc_to_tensor(preproc_u8):
    """(256, 256, 3) uint8 -> (1, 3, 256, 256) float32 ImageNet-normalised."""
    f = preproc_u8.astype(np.float32)
    f = (f - _IMG_MEAN) / _IMG_STD
    chw = f.transpose(2, 0, 1)
    return np.ascontiguousarray(chw[None, ...])  # (1, 3, 256, 256)


def decode_simcc(simcc_x, simcc_y, simcc_split=2.0):
    """(1, K, W) -> (K, 2) xy pixels in model space."""
    x_idx = simcc_x[0].argmax(axis=-1).astype(np.float32) / simcc_split
    y_idx = simcc_y[0].argmax(axis=-1).astype(np.float32) / simcc_split
    return np.stack([x_idx, y_idx], axis=-1)  # (K, 2)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    npz_path, onnx_path = sys.argv[1], sys.argv[2]

    d = np.load(npz_path)
    preproc = d["preproc_u8"]  # (N, 256, 256, 3)
    print(f"Loaded {preproc.shape[0]} preproc frames from {npz_path}")
    print(f"Loading ONNX session from {onnx_path}")

    sess = make_session(onnx_path, provider="cpu")
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print(f"Inputs: {in_name}  Outputs: {out_names}")

    # Track landmark 0 and 32 (the jaw endpoints that flip)
    LOOK = [0, 4, 16, 32, 36, 42, 96, 100]

    def run_one(tensor):
        outs = sess.run(out_names, {in_name: tensor})
        simcc_x, simcc_y = outs[0], outs[1]
        return decode_simcc(simcc_x, simcc_y)

    def pretty(kpts, label):
        print(f"\n{label}")
        print(f"  {'idx':>3} {'x':>8} {'y':>8}")
        for idx in LOOK:
            x, y = kpts[idx]
            print(f"  {idx:3d} {x:8.2f} {y:8.2f}")

    # =============================================================
    # Scenario 1: feed the SAME f0 preproc 8 times
    # =============================================================
    print("\n" + "=" * 70)
    print("SCENARIO 1: feed f0.preproc 8 times to the SAME session")
    print("  -> outputs MUST be bit-identical if session is stateless")
    print("=" * 70)
    t0 = preproc_to_tensor(preproc[0])
    ref = None
    all_identical = True
    for i in range(8):
        kp = run_one(t0)
        if ref is None:
            ref = kp
            pretty(kp, f"run {i} (reference)")
        else:
            diff = np.abs(kp - ref).max()
            is_same = diff < 1e-3
            status = "OK" if is_same else "DRIFT"
            print(f"run {i}: max_abs_diff_vs_ref = {diff:.4f}  [{status}]")
            if not is_same:
                all_identical = False
                pretty(kp, f"run {i} (drifted)")
    if all_identical:
        print("\n  VERDICT: session is stateless on identical input.")
        print("  => Bug is NOT session-state pollution; it's input-sensitive.")
    else:
        print("\n  VERDICT: session OUTPUT DRIFTS on identical input.")
        print("  => Session has hidden state. We need fresh sessions per call.")

    # =============================================================
    # Scenario 2: real sequence, same session
    # =============================================================
    print("\n" + "=" * 70)
    print("SCENARIO 2: feed real f0, f1, f2, f3 sequentially (same session)")
    print("  -> expected to reproduce the alternating bug")
    print("=" * 70)
    for f in range(preproc.shape[0]):
        tf = preproc_to_tensor(preproc[f])
        kp = run_one(tf)
        pretty(kp, f"frame {f}")

    # =============================================================
    # Scenario 3: real sequence, FRESH session per frame
    # =============================================================
    print("\n" + "=" * 70)
    print("SCENARIO 3: feed real f0, f1, f2, f3 each with a FRESH session")
    print("  -> if SCENARIO 2 alternated but this doesn't, session reuse is the bug")
    print("=" * 70)
    for f in range(preproc.shape[0]):
        fresh = make_session(onnx_path, provider="cpu")
        fin = fresh.get_inputs()[0].name
        fout = [o.name for o in fresh.get_outputs()]
        tf = preproc_to_tensor(preproc[f])
        outs = fresh.run(fout, {fin: tf})
        kp = decode_simcc(outs[0], outs[1])
        pretty(kp, f"frame {f} (fresh session)")


if __name__ == "__main__":
    main()
