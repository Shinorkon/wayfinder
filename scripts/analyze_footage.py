#!/usr/bin/env python3
"""
Wayfinder offline analysis pipeline.

Runs the real perception stack over every clip in data/footage and writes:

  data/analysis/<clip>.json      measured stop telemetry (arrival rate, queue, dwell)
  data/analysis/index.json       per-stop view of that telemetry + provenance
  output/keyframes/<clip>/*.jpg  raw / detections / segmentation / hybrid renders

These artifacts are the sole source of arrival-rate numbers in the dashboard. Run it
whenever footage changes:

    python3 scripts/analyze_footage.py                 # everything, including SAM 3
    python3 scripts/analyze_footage.py --skip-sam3     # faster, no segmentation renders
    python3 scripts/analyze_footage.py --only 01       # one clip by filename prefix
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv.detector import PersonDetector                      # noqa: E402
from src.cv.keyframes import load_sources, render_keyframes      # noqa: E402
from src.cv.stop_analysis import analyse_clip, to_dict  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOOTAGE_DIR = os.path.join(ROOT, "data", "footage")
ANALYSIS_DIR = os.path.join(ROOT, "data", "analysis")
KEYFRAME_DIR = os.path.join(ROOT, "output", "keyframes")
NETWORK_PATH = os.path.join(ROOT, "data", "routes", "male_rtl_network.json")


def build_index(
    clip_artifacts: Dict[str, Dict[str, Any]],
    keyframes: Dict[str, Dict[str, Any]],
    network: Dict[str, Any],
) -> Dict[str, Any]:
    """Joins measured clip telemetry onto network stops."""
    stops_out: List[Dict[str, Any]] = []
    for stop in network["stops"]:
        clip = stop.get("video_filename")
        art = clip_artifacts.get(clip) if clip else None
        measured = None
        provenance = None
        if art:
            measured = {
                "arrival_rate_pax_per_min": art["metrics"]["arrival_rate_pax_per_min"],
                "arrival_rate_zone_entries_pax_per_min": art["metrics"][
                    "arrival_rate_zone_entries_pax_per_min"
                ],
                "queue_mean_pax": art["metrics"]["queue_mean_pax"],
                "queue_p95_pax": art["metrics"]["queue_p95_pax"],
                "queue_peak_pax": art["metrics"]["queue_peak_pax"],
                "mean_dwell_s": art["metrics"]["mean_dwell_s"],
                "waiting_tracks": art["metrics"]["waiting_tracks"],
                "unique_tracks": art["metrics"]["unique_tracks"],
                "analysed_span_s": art["analysis"]["analysed_span_s"],
                "confidence": art["confidence"]["grade"],
                "confidence_reasons": art["confidence"]["reasons"],
                "queue_series": art["queue_series"],
                "limitations": art["limitations"],
                "analysis_method": art["analysis"]["method"],
                # The waiting-zone raster is stored so the live monitor reuses the exact
                # region the offline measurement used, instead of deriving its own.
                "waiting_zone": art.get("waiting_zone", {}),
            }
            provenance = {
                "clip": clip,
                "title": (art.get("sources") or {}).get("title"),
                "city": (art.get("sources") or {}).get("city"),
                "source": (art.get("sources") or {}).get("source"),
                "license": (art.get("sources") or {}).get("license"),
                "disclosure": (art.get("sources") or {}).get("disclosure"),
                "redistributable": (art.get("sources") or {}).get("redistributable"),
            }
        kf = keyframes.get(clip, {}) if clip else {}
        # Expose server-absolute paths so the client never has to guess the mount point.
        keyframe_urls = {
            layer: f"/output/keyframes/{rel}" for layer, rel in (kf.get("images") or {}).items()
        }
        stops_out.append(
            {
                **{k: v for k, v in stop.items() if not k.startswith("sample_keyframe_")},
                "measured": measured,
                "footage_provenance": provenance,
                "keyframes": keyframe_urls,
                "keyframe_timestamp_s": kf.get("timestamp_s"),
                "detections_in_keyframe": kf.get("detections_in_frame"),
                "segmentation": kf.get("segmentation", {"available": False}),
            }
        )

    measured_stops = [s for s in stops_out if s["measured"]]
    return {
        "network": network.get("network"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "scripts/analyze_footage.py",
        "coverage": {
            "stops_total": len(stops_out),
            "stops_with_measured_footage": len(measured_stops),
            "stops_without_footage": [s["id"] for s in stops_out if not s["measured"]],
            "confidence_counts": {
                g: sum(
                    1
                    for s in measured_stops
                    if s["measured"]["confidence"] == g
                )
                for g in ("high", "medium", "low")
            },
        },
        "clips": clip_artifacts,
        "stops": stops_out,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Wayfinder offline footage analysis")
    ap.add_argument("--skip-sam3", action="store_true", help="skip SAM 3 segmentation renders")
    ap.add_argument("--only", default=None, help="only clips whose filename starts with this")
    ap.add_argument("--max-seconds", type=float, default=None, help="override analysis window")
    args = ap.parse_args()

    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(KEYFRAME_DIR, exist_ok=True)

    clips = sorted(glob.glob(os.path.join(FOOTAGE_DIR, "*.mp4")))
    if args.only:
        clips = [c for c in clips if os.path.basename(c).startswith(args.only)]
    if not clips:
        print("no clips found", file=sys.stderr)
        return 1

    sources = load_sources(os.path.join(FOOTAGE_DIR, "SOURCES.json"))
    with open(NETWORK_PATH, "r", encoding="utf-8") as f:
        network = json.load(f)

    detector = PersonDetector()
    sam3 = None
    if not args.skip_sam3:
        from src.cv.sam3_masker import SAM3Masker

        sam3 = SAM3Masker.get_instance()

    clip_artifacts: Dict[str, Dict[str, Any]] = {}
    keyframes: Dict[str, Dict[str, Any]] = {}

    for clip_path in clips:
        name = os.path.basename(clip_path)
        print(f"\n=== {name}")
        t0 = time.time()
        params = {"max_analysis_seconds": args.max_seconds} if args.max_seconds else None
        analysis = analyse_clip(clip_path, detector, params=params)
        art = to_dict(analysis)
        art["sources"] = sources["clips"].get(name, {})
        clip_artifacts[name] = art
        # Write the same payload we index, so each per-clip artifact documents its own
        # provenance and can be read standalone.
        os.makedirs(ANALYSIS_DIR, exist_ok=True)
        with open(os.path.join(ANALYSIS_DIR, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(art, fh, indent=2)

        m = art["metrics"]
        print(
            f"    measured: inflow {m['arrival_rate_pax_per_min']} pax/min "
            f"(zone entries {m['arrival_rate_zone_entries_pax_per_min']}), "
            f"queue mean {m['queue_mean_pax']} / p95 {m['queue_p95_pax']}, "
            f"dwell {m['mean_dwell_s']}s, confidence {art['confidence']['grade']}"
        )

        kf = render_keyframes(
            clip_path,
            os.path.join(KEYFRAME_DIR, os.path.splitext(name)[0]),
            detector,
            sam3=sam3,
            analysis=art,
            source_meta=sources["clips"],
        )
        keyframes[name] = kf
        print(f"    rendered {len(kf['images'])} layers in {time.time() - t0:.1f}s")

    index = build_index(clip_artifacts, keyframes, network)
    index_path = os.path.join(ANALYSIS_DIR, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    cov = index["coverage"]
    print(
        f"\nwrote {index_path}\n"
        f"  stops with measured footage: {cov['stops_with_measured_footage']}/{cov['stops_total']}\n"
        f"  confidence: {cov['confidence_counts']}\n"
        f"  stops without footage: {cov['stops_without_footage']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
