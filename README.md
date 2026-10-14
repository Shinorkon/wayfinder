# 🧭 Wayfinder

**Stop-level demand measurement and bus frequency modelling for the Greater Malé RTL network.**

Wayfinder measures how many people are waiting at a bus stop — and how fast they arrive —
from fixed-camera footage, then uses that measurement to size bus headways and fleet
requirements with an explicit queueing model.

It is a datathon prototype. It is **not** an operational dispatch system, and no official
RTL demand, fleet or timetable data is used anywhere in it. Every number on the dashboard
is labelled as *measured*, *simulated* or *modelled*.

---

## What is actually measured

The perception stack runs offline over each clip in `data/footage/`:

| Stage | Module | What it produces |
|---|---|---|
| Per-frame detection + masks | `src/cv/segmenter.py` | YOLOv8n-seg person boxes **and** one mask per person, from a single forward pass |
| Temporal association | `src/cv/tracker.py` | Track identity across frames (IoU + constant-velocity, scipy Hungarian assignment) |
| Waiting-zone derivation | `src/cv/stop_analysis.py` | The region where tracked people loiter (dwell-density heatmap) |
| Measurement | `src/cv/stop_analysis.py` | Arrival rate (pax/min), standing queue, dwell time, time series |

### Why the live path uses a YOLO segmentation model, not SAM 3

Measured on this machine's CPU (540p frames, `imgsz=640`, person class only):

| Model | ms/frame | Effective fps | Weights |
|---|---|---|---|
| **yolov8n-seg** (used) | **39** | **25.5** | 6.7 MB |
| yolo11n-seg | 112 | 8.9 | 5.9 MB |
| yolov8s-seg | 237 | 4.2 | 22.8 MB |
| SAM 3 (1008×1008) | ~14 000 | 0.07 | 3.4 GB |

SAM 3 is the stronger model, but it needs roughly 14 seconds per frame here, accepts only
1008×1008 input, and therefore can never drive a live view. A YOLO segmentation network
returns the boxes *and* the masks in the same forward pass, which is exactly what this
application needs — and in side-by-side comparison its masks are tighter on people
(SAM 3 bleeds onto bus windows and handrails at the same clip).

There is a measurement consequence worth stating: because yolov8n-seg does not detect
people seen through bus windows or glass, it reports *fewer* waiting passengers than the
boxes-only model on the same clip (24.3 vs 40.5 pax/min on clip 01). Both verdicts are
identical — low confidence, schedule held — but the segmentation model's numbers are the
more conservative ones, which is the right direction for a dispatch decision.

SAM 3 remains available as an optional slow "deep pass"
(`POST /api/live/start` with `deep_pass_interval_s > 0`, or
`scripts/analyze_footage.py --sam3-deep-pass` for keyframes) because it genuinely
segments ambiguous objects better. It is simply not the default.

Arrival rate is counted as **distinct tracks that entered the derived waiting zone and then
remained stationary long enough to be considered waiting**. It is not a per-frame head count,
and it cannot be obtained from a single image — that is why the tracker exists.

Every measurement ships with a **confidence grade**. If a clip is too short, too sparse or
too fragmented for a stable rate, the grade is `low`, and the dispatch model then **holds the
existing timetable** instead of compressing headway on weak evidence.

### Honest limitations, measured on this repository's own footage

| Clip | Analysed | Waiting tracks | Mean dwell | Measured inflow | Confidence |
|---|---|---|---|---|---|
| 01 Queen Square bus station (Liverpool) | 14.8 s | 9 | 6.9 s | 40.5 pax/min | low — window under 45 s |
| 02 Elevated street, night | 35.0 s | 7 | 5.7 s | 13.7 pax/min | low — window under 45 s |
| 03 Station concourse | 17.2 s | 12 | 2.6 s | 45.4 pax/min | low — dwell too short |
| 04 Station concourse, growing | 111.4 s | 14 | 3.1 s | 9.2 pax/min | **medium** |
| 05 Outdoor plaza crowd | 96.2 s | 15 | 2.5 s | 8.7 pax/min | low — dwell too short |

The conclusion this data supports: **these clips are good enough to demonstrate the
measurement pipeline and not good enough to dispatch a real bus.** Clips of 10–20 seconds
cannot estimate an arrival process, and dense low-resolution crowds fragment track identity,
which inflates counts. The dashboard says this on screen rather than hiding it.

---

## What is modelled

`src/intelligence/dispatch_engine.py` implements one explicit model:

```
measurement      lambda = arrival rate (pax/min, measured)
                 q0     = standing queue (pax, measured)

service          P  = seats x (1 - reserve_fraction)      planned usable capacity
                 H  = headway (min)
                 mu = P / H                               boarding rate (pax/min)

bottleneck       net = mu - lambda       net <= 0  ->  queue cannot be cleared (reported as such)

wait             E[W] = H/2 x (1 + CV^2)                  Welch's formula
clearance        T = q0 / net                             None when net <= 0

fleet            buses = ceil(round_trip / H), capped by vehicles that exist
```

Design rules that the tests enforce:

* Baseline and recommended wait use the **same** formula, so a reported improvement is not circular.
* Extending a headway reports a **negative** improvement and labels it an energy trade-off.
* A queue that cannot be cleared reports `null`, never a comfortable number.
* Reducing headway can never reduce the required bus count.
* Fleet requirements are capped by real vehicle counts; a shortfall is reported, not silently satisfied.

---

## Honesty rules (enforced by tests)

`tests/test_data_integrity.py` fails the build if these are violated:

1. **No randomness in the telemetry path.** `random.randint` is rejected by AST inspection of
   `src/intelligence/telemetry.py` and `src/api/server.py`.
2. **No invented operational detail.** Strings like fake depots, vehicle registrations and
   hardcoded delay figures are rejected.
3. **Provenance is mandatory.** Scenarios declare a multiplier; the unscaled base rate stays
   visible in the API response next to the scaled value.

Before the overhaul this project generated passenger counts with `random.randint()`, labelled
Liverpool bus-stop footage as Maldivian CCTV, and printed invented dispatch directives naming
a depot and a bus that do not exist. Those specific failure modes now have tests.

---

## Repository layout

```
wayfinder/
├── data/
│   ├── footage/                 # public reference clips + SOURCES.json (licences, disclosure)
│   ├── analysis/                # generated: per-clip measurements + index.json
│   └── routes/                  # Greater Malé network, OSM road geometry, fleet config
├── src/
│   ├── cv/
│   │   ├── segmenter.py         # YOLOv8n-seg: boxes + masks in one pass (live path)
│   │   ├── detector.py          # YOLOv8n boxes-only path
│   │   ├── tracker.py           # IoU + constant-velocity multi-object tracker
│   │   ├── stop_analysis.py     # measurement: arrival rate, queue, dwell, confidence
│   │   ├── keyframes.py         # raw / detections / segmentation / hybrid renders
│   │   └── sam3_masker.py       # optional SAM 3 deep pass (slow, off by default)
│   ├── intelligence/
│   │   ├── telemetry.py         # single source of truth for dashboard numbers
│   │   ├── dispatch_engine.py   # the queueing model above
│   │   └── deepseek_agent.py    # optional advisory text, grounded in computed values
│   └── api/
│       ├── server.py            # FastAPI app
│       └── router.py            # OSRM road-routing helper (used offline to build geometry)
├── web/                         # dashboard (vanilla JS, Leaflet, no hardcoded figures)
├── scripts/
│   ├── analyze_footage.py       # offline pipeline: footage -> measurements -> keyframes
│   └── verify_pipeline.py       # end-to-end verification with real assertions
├── tests/                       # pytest suite (integrity, engine invariants, API, CV)
├── requirements.txt
├── run.sh
├── LICENSE
└── THIRD_PARTY.md
```

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Analyse the footage (writes data/analysis/ and output/keyframes/).
python3 scripts/analyze_footage.py                  # ~15 s total, YOLOv8n-seg masks
#    or, to render the mask layer with SAM 3 instead (about 40 s per clip):
python3 scripts/analyze_footage.py --sam3-deep-pass

# 2. Verify everything end to end.
python3 -m pytest tests/ -q
python3 scripts/verify_pipeline.py

# 3. Launch the dashboard.
./run.sh            # http://localhost:8000
```

`run.sh` sets its own directory and uses whatever `python3` is on your `PATH` inside the
activated virtualenv. It does not hardcode anyone's home directory.

### Live monitoring

The dashboard streams a stop's camera through a single worker that runs the segmentation
model on every sampled frame (default 5 fps) and feeds a tracker, so counts and masks come
from the same pass and can never disagree. Measured on CPU: 80–140 ms per frame for 1080p
input, holding 5 fps. One stop is monitored at a time — that is what this hardware supports,
and the API says so rather than pretending otherwise.

### GPU note

Everything runs on CPU if CUDA is unavailable, and reports the device it actually used.
No part of this project claims GPU acceleration it is not getting.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness and whether analysis artefacts are loaded |
| `GET /api/meta` | Method, limitations, scenarios, coverage, disclaimer |
| `GET /api/network` | Stops and routes with real OSM geometry |
| `GET /api/routes/{id}/geometry` | One route's road path and schedule parameters |
| `GET /api/stops` | All stops: measured demand, provenance, computed dispatch |
| `GET /api/stops/{id}` | One stop |
| `GET /api/telemetry` | Network KPIs, each labelled measured/simulated/modelled |
| `POST /api/scenario` | Switch demand scenario (declared multipliers applied to measured base rates) |
| `POST /api/advisory` | Optional DeepSeek commentary; numbers always come from the model |

---

## Provenance of the footage

The clips in `data/footage/` are **public reference videos**, not RTL cameras. They are from
Liverpool, UK and from the MOTChallenge research datasets, and they are of other cities.
`data/footage/SOURCES.json` records the source, licence and a disclosure string for each, and
that disclosure is burned into every rendered keyframe and shown in the dashboard.

MOT17/MOT20 clips are **CC BY-NC-SA 3.0 — non-commercial, no redistribution**. Do not ship
them in a commercial product or a public repository. See `THIRD_PARTY.md`.

---

## Licence

Project code: MIT (see `LICENSE`). Bundled footage and model weights keep their own licences.
