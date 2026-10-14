# Third-party notices

Wayfinder's own source code is MIT licensed (see `LICENSE`). This file records the
third-party material that the project uses or bundles, so nobody redistributes something
they are not allowed to.

## Bundled footage — `data/footage/`

These clips are **public reference videos**. They are not RTL cameras, they are not from
the Maldives, and they are used only to exercise and demonstrate the computer-vision
pipeline. The machine-readable version of this table is `data/footage/SOURCES.json`, and
the disclosure string from it is burned into every rendered keyframe.

| File | Source | Licence | Redistributable |
|---|---|---|---|
| `01_bus_stop_people_pexels_1601542_1080p.mp4` | [Pexels](https://www.pexels.com/video/people-at-a-bus-stop-1601542/) — Queen Square bus station, Liverpool, UK | [Pexels License](https://www.pexels.com/license/) | Yes |
| `02_cctv_elevated_street_night_MOT17-04_540p.mp4` | [MOTChallenge MOT17-04](https://motchallenge.net/vis/MOT17-04-SDP) preview | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | **No** |
| `03_cctv_station_concourse_MOT20-01_540p.mp4` | [MOTChallenge MOT20-01](https://motchallenge.net/vis/MOT20-01) preview | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | **No** |
| `04_cctv_station_concourse_MOT20-02_540p.mp4` | [MOTChallenge MOT20-02](https://motchallenge.net/vis/MOT20-02) preview | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | **No** |
| `05_cctv_outdoor_plaza_crowd_MOT20-03_440p.mp4` | [MOTChallenge MOT20-03](https://motchallenge.net/vis/MOT20-03) preview | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | **No** |

### What CC BY-NC-SA 3.0 means here

* **BY** — attribution required. Cite:
  * Milan, A., Leal-Taixé, L., Reid, I., Roth, S., Schindler, K. (2016). *MOT16: A Benchmark for Multi-Object Tracking.*
  * Dendorfer, P., et al. (2020). *MOT20: A Benchmark for Multi Object Tracking in Crowded Scenes.*
* **NC** — non-commercial use only. Do not use these clips in a commercial product or service.
* **SA** — derivative works must carry the same licence.

The MOT clips are included as a **local evaluation aid for this prototype**. Before any
public or commercial release, delete `data/footage/02_*` through `05_*` and regenerate the
analysis:

```bash
rm data/footage/0[2-5]_*.mp4
rm -rf data/analysis output/keyframes
python3 scripts/analyze_footage.py
```

Stops whose footage has been removed will report **"No analysed footage for this stop"**
and will issue no headway recommendation. That is the intended behaviour: the system
reports missing data instead of inventing it.

## Models

| Component | Licence | Notes |
|---|---|---|
| YOLOv8n (`yolov8n.pt`) | AGPL-3.0 (Ultralytics) | Not committed to this repository. AGPL is a strong copyleft licence — review it before any commercial deployment. |
| SAM 3 weights (`~/.config/lightcraft/models/sam3`) | Meta SAM 3 licence | Not committed. Referenced by path only; the pipeline runs without them if `--skip-sam3` is used. |

## Python dependencies

See `requirements.txt`. All are permissively licensed except where noted above
(`ultralytics` is AGPL-3.0).

## Basemap and routing data

* Map tiles: © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, ODbL.
* Route geometry: computed offline with the [OSRM](http://project-osrm.org/) demo server
  against OpenStreetMap road data, then stored in `data/routes/male_rtl_network.json`.
  OpenStreetMap data is ODbL — attribution is required if you redistribute the geometry.

## Network data

Stop coordinates are public map coordinates, hand-checked against OpenStreetMap. Fleet
composition, headway parameters and capacity thresholds are **documented assumptions**, not
official RTL data; `data/routes/male_rtl_network.json` says so in its `data_status` block.
