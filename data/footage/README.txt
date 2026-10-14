Datathon test footage: people counting and SAM 3 segmentation
Updated 2026-10-14. Clip 01 is from Pexels. Clips 02-05 are the official MOTChallenge
sequence preview videos (webm), converted to H.264 mp4 at their original resolution.
Every clip was checked with ffprobe and by looking at frames: fixed camera, multiple people.

LICENSES
- Pexels: Pexels License (free to use, including commercially; no attribution needed).
- MOTChallenge (MOT17 / MOT20): CC BY-NC-SA 3.0. Non-commercial research and testing only.
  Cite: Milan et al. 2016 (MOT16/17); Dendorfer et al. 2020 (MOT20).
- Clips 02-05 have a burned-in "MOTxx-yy: RAW" label and a frame counter, like a CCTV overlay.
- Their resolution is low (960x540 or 586x440) because these are the site preview renders.
  Full-resolution frames are only in the full dataset zips (MOT17 ~5.5GB, MOT20 ~5GB).

1) 01_bus_stop_people_pexels_1601542_1080p.mp4
   Source: https://www.pexels.com/video/people-at-a-bus-stop-1601542/
   ~15 s, 1920x1080, 25fps. Pexels License.
   Why: Fixed view of a city bus stop with several people waiting and buses arriving. Good for counting people in a waiting zone.

2) 02_cctv_elevated_street_night_MOT17-04_540p.mp4
   Source: https://motchallenge.net/vis/MOT17-04-SDP (video: https://motchallenge.net/sequenceVideos/MOT17-04-SDP-raw.webm)
   ~35 s, 960x540, 30fps. CC BY-NC-SA 3.0.
   Why: Classic static, elevated security-camera view of a pedestrian street at night with about 15-20 people. Whole bodies visible, little occlusion.

3) 03_cctv_station_concourse_MOT20-01_540p.mp4
   Source: https://motchallenge.net/vis/MOT20-01 (video: https://motchallenge.net/sequenceVideos/MOT20-01-raw.webm)
   ~17 s, 960x540, 25fps. CC BY-NC-SA 3.0.
   Why: High fixed camera over a busy train-station hall with about 30-50 people. Good for counting crowds of moderate density.

4) 04_cctv_station_concourse_MOT20-02_540p.mp4
   Source: https://motchallenge.net/vis/MOT20-02 (video: https://motchallenge.net/sequenceVideos/MOT20-02-raw.webm)
   ~111 s, 960x540, 25fps. CC BY-NC-SA 3.0.
   Why: Same kind of elevated station-hall view. The crowd grows from about 15 to 40+ people, which tests how counts and masks hold up as the scene gets denser.

5) 05_cctv_outdoor_plaza_crowd_MOT20-03_440p.mp4
   Source: https://motchallenge.net/vis/MOT20-03 (video: https://motchallenge.net/sequenceVideos/MOT20-03-raw.webm)
   ~96 s, 586x440, 25fps. CC BY-NC-SA 3.0.
   Why: High-angle CCTV view of a crowded outdoor plaza near a crosswalk at night. Many small people, which stresses dense counting and segmenting small figures.

Total: about 4.6 minutes.
