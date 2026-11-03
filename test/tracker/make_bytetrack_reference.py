"""Record the ByteTrack reference outputs used by test_byte_tracker_equivalence.py.

Run it ONLY with the tracker whose behavior is the reference (fork 0dc53f92, before
the speed-up); the speed-up must reproduce these outputs.

    python -m test.tracker.make_bytetrack_reference <dir with recorded .npz>

Writes:
- test/tracker/data/bytetrack_reference.npz (committed): slices of the recordings and
  synthetic sequences, with the tracker id of every detection per configuration.
- <dir>/<name>.ref.npz (not committed): full-length tracker ids and output track
  states (external id + Kalman tlbr per frame), checked when the test runs with
  BYTETRACK_REFERENCE_DIR=<dir>.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from test.tracker.bytetrack_replay import CONFIGS, replay, synthetic

# Recordings from Ball Counter v2 (tracker input per frame, conf >= 0.5):
# dense flow (221, 249, 253 balls/min class videos) and the sparse rg0001 line.
SLICES = {
    "l221": ("20261029_13_linea221.npz", 1500),
    "l253": ("20261029_12_linea253.npz", 1500),
    "rg0001": ("20261014_16_rg0001_molycom_04.npz", 3000),
}
SYNTHETIC_SEEDS = (0, 1, 2)
OUT = Path(__file__).parent / "data" / "bytetrack_reference.npz"


def main(source_dir: Path) -> None:
    arrays = {}
    for name, (file, n) in SLICES.items():
        z = np.load(source_dir / file)
        keep = z["frame"] <= n
        arrays[f"{name}/frame"] = z["frame"][keep]
        arrays[f"{name}/xyxy"] = z["xyxy"][keep]
        arrays[f"{name}/conf"] = z["conf"][keep]
        arrays[f"{name}/n_frames"] = np.int32(n)
    for seed in SYNTHETIC_SEEDS:
        frame, xyxy, conf, n = synthetic(seed)
        name = f"synthetic{seed}"
        arrays.update(
            {
                f"{name}/frame": frame,
                f"{name}/xyxy": xyxy,
                f"{name}/conf": conf,
                f"{name}/n_frames": np.int32(n),
            }
        )
    names = list(SLICES) + [f"synthetic{s}" for s in SYNTHETIC_SEEDS]
    for name in names:
        for config, params in CONFIGS.items():
            arrays[f"{name}/{config}/tracker_id"] = replay(
                arrays[f"{name}/frame"],
                arrays[f"{name}/xyxy"],
                arrays[f"{name}/conf"],
                int(arrays[f"{name}/n_frames"]),
                params,
            )
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"{OUT}: {OUT.stat().st_size / 1e3:.0f} kB")

    for file in sorted(source_dir.glob("*.npz")):
        if file.name.endswith(".ref.npz"):
            continue
        z = np.load(file)
        ref = {}
        for config, params in CONFIGS.items():
            tid, sf, sid, stlbr = replay(
                z["frame"],
                z["xyxy"],
                z["conf"],
                int(z["n_frames"]),
                params,
                with_states=True,
            )
            ref.update(
                {
                    f"{config}/tracker_id": tid,
                    f"{config}/state_frame": sf,
                    f"{config}/state_id": sid,
                    f"{config}/state_tlbr": stlbr,
                }
            )
        target = file.with_name(file.stem + ".ref.npz")
        np.savez_compressed(target, **ref)
        print(f"{target}: {int(z['n_frames'])} frames, {len(z['frame'])} detections")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
