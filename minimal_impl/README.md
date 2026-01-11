Minimal DP-SVT + Gated Attention (toy)

This is a tiny, self-contained example that demonstrates:
- Gated attention (soft gating on attention output)
- SVT-based selection of parameter blocks
- DP-SGD style per-sample clipping and Gaussian noise

Run (CPU only):
  python minimal_impl/minimal_svt_gated_dp.py --steps 30 --verify

Expected output:
- Training runs without error
- "selected_blocks" never exceeds the SVT cap
- Gate values stay in [0, 1]

Notes:
- This is a toy example for verification and debugging, not a production DP guarantee.
- The privacy accounting printed is a conservative Gaussian bound (no subsampling).
- Per-sample gradients are computed in a loop for clarity, not performance.
