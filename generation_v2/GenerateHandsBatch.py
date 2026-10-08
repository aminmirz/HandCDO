from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from datetime import datetime

from Config import HandConfig
from HandClass import Hand
from ParamGen import FingerParams, PalmParams


palm_range_overrides = {
    "initial_finger_number": (1, 4),
    "initial_thumb_number": 1,
    "initial_outline_sides": (4, 6),
    "initial_outline_rotation_deg": 45.0,
    "initial_resolution_mm": 1.0,
    "initial_smoothing_iters": 6,
    "initial_smoothing_t": 0.25,
    "initial_bump_type": "gaussian",
    "initial_bump_number": (0, 3),
    "initial_bump_max_height_intensity_mm": (0.0, 15.0),
    "initial_bump_height_intensity": (0.0, 1.0),
    "initial_bumps_spread": (0.1, 0.5),
    "initial_bumps_aspect_ratio": (1.0, 3.0),
    "initial_bump_rotation_deg": (0, 180),
    "initial_bump_center_angle_deg": (0.0, 360.0),
    "initial_bump_center_offset": (0.0, 1.0),
}


def _write_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def generate_one(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    palm_cfg = PalmParams(fix=False, thumb_side="right", **palm_range_overrides).build()
    palm_cfg.pad_resolution_level = 5
    palm_cfg.save_config(os.path.join(output_dir, "palm_cfg.py"))

    finger_cfgs = [FingerParams("finger", fix=False).build() for _ in range(palm_cfg.finger_number)]
    for finger_cfg in finger_cfgs:
        finger_cfg.pad_resolution_level = 4
    thumb_cfgs = [FingerParams("thumb", fix=False).build() for _ in range(palm_cfg.thumb_number)]

    for idx, finger_cfg in enumerate(finger_cfgs):
        finger_cfg.save_config(os.path.join(output_dir, f"finger_cfg_{idx}.py"))
    for idx, thumb_cfg in enumerate(thumb_cfgs):
        thumb_cfg.save_config(os.path.join(output_dir, f"thumb_cfg_{idx}.py"))

    hand_cfg = HandConfig(data="fixed")
    hand_cfg.collision_mesh = False
    hand_cfg.save_config(os.path.join(output_dir, "hand_cfg.py"))

    hand = Hand(
        palm_cfg=palm_cfg,
        finger_cfgs=finger_cfgs,
        thumb_cfgs=thumb_cfgs,
        hand_cfg=hand_cfg,
        root_dir=output_dir,
    )
    hand.blender_full_assembly()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_successes", type=int, default=50)
    parser.add_argument("--max_attempts", type=int, default=80)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean and os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": os.path.abspath(args.output_dir),
        "target_successes": args.target_successes,
        "max_attempts": args.max_attempts,
        "successes": [],
        "failures": [],
    }
    summary_path = os.path.join(args.output_dir, "batch_summary.json")

    attempt = 0
    success_count = 0
    while attempt < args.max_attempts and success_count < args.target_successes:
        hand_name = f"hand_{success_count:03d}" if success_count < args.target_successes else f"extra_{attempt:03d}"
        attempt_dir = os.path.join(args.output_dir, hand_name)
        if os.path.exists(attempt_dir):
            shutil.rmtree(attempt_dir)

        record = {"attempt": attempt, "directory": hand_name}
        print(f"[batch] attempt {attempt + 1}/{args.max_attempts}: {hand_name}", flush=True)
        try:
            generate_one(attempt_dir)
            record["status"] = "success"
            summary["successes"].append(record)
            success_count += 1
            print(f"[batch] success {success_count}/{args.target_successes}: {hand_name}", flush=True)
        except Exception as exc:
            failed_name = f"failed_attempt_{attempt:03d}"
            failed_dir = os.path.join(args.output_dir, failed_name)
            if os.path.exists(failed_dir):
                shutil.rmtree(failed_dir)
            if os.path.exists(attempt_dir):
                os.replace(attempt_dir, failed_dir)
            record.update(
                {
                    "status": "failed",
                    "directory": failed_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            summary["failures"].append(record)
            with open(os.path.join(failed_dir, "failure.txt"), "w", encoding="utf-8") as f:
                f.write(record["traceback"])
            print(f"[batch] failed attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
        finally:
            attempt += 1
            summary["attempts_completed"] = attempt
            summary["success_count"] = success_count
            summary["failure_count"] = len(summary["failures"])
            summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
            _write_json(summary_path, summary)

    return 0 if success_count >= args.target_successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
