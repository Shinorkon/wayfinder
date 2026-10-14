/**
 * Wayfinder live dashboard.
 *
 * Deliberately small. One camera, three numbers, one sentence.
 *
 * How the live view works: the backend continuously monitors the selected stop and
 * publishes detections with normalised coordinates. This file draws those boxes and
 * masks onto a canvas laid over the playing video. The video element plays the clip
 * itself, so the picture is smooth; the box overlay refreshes a few times a second
 * and therefore trails the picture slightly. That lag is shown to the user rather
 * than hidden.
 */

const API = {
  meta: '/api/meta',
  stops: '/api/stops',
  live: (id) => `/api/live/${id}`,
  liveStart: '/api/live/start',
  liveStop: '/api/live/stop',
  scenario: '/api/scenario',
};

const state = {
  meta: null,
  stops: [],
  stopId: null,
  mode: 'live',
  monitoring: false,
  overlay: null,
  map: null,
  markers: {},
  timers: { overlay: null, stop: null },
  lastWaiting: 0,
  lastWaitingAt: 0,
};

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ helpers

async function getJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.json();
}

const num = (value, digits = 0) =>
  value === null || value === undefined || Number.isNaN(value) ? '–' : Number(value).toFixed(digits);

// --------------------------------------------------------------------- boot

async function boot() {
  try {
    state.meta = await getJson(API.meta);
  } catch (error) {
    $('sentence').textContent =
      `Cannot reach the Wayfinder server (${error.message}). Start it with ./run.sh`;
    return;
  }

  buildScenarioSelect();
  buildAbout();
  wireControls();

  const payload = await getJson(API.stops);
  state.stops = payload.stops;
  initMap();
  renderStops();
  // Poll unconditionally from boot: the live state is what this screen is about, and
  // starting the interval only on a mode switch left the first view frozen.
  startPolling();
  selectStop(state.stops[0].id);
}

function buildScenarioSelect() {
  const select = $('scenario');
  select.innerHTML = '';
  state.meta.scenarios.forEach((scenario) => {
    const option = document.createElement('option');
    option.value = scenario.key;
    option.textContent = scenario.simulated ? `${scenario.label} (simulated)` : scenario.label;
    if (scenario.key === state.meta.current_scenario.key) option.selected = true;
    select.appendChild(option);
  });
}

function wireControls() {
  $('scenario').addEventListener('change', (event) => setScenario(event.target.value));
  $('mode-live').addEventListener('click', () => setMode('live'));
  $('mode-snap').addEventListener('click', () => setMode('snap'));
  const openAbout = () => $('about').showModal();
  $('info-btn').addEventListener('click', openAbout);
  $('about-btn').addEventListener('click', openAbout);
}

// ---------------------------------------------------------------------- map

function initMap() {
  state.map = L.map('map', { center: [4.2035, 73.5255], zoom: 12, zoomControl: true });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors',
  }).addTo(state.map);

  const network = state.meta.network_geometry || [];
  network.forEach((route) => {
    const points = route.osrm_geometry || route.waypoints || [];
    if (!points.length) return;
    L.polyline(points, { color: '#ffffff', weight: 5, opacity: 0.55 }).addTo(state.map);
    L.polyline(points, { color: route.color || '#4ea3f5', weight: 3, opacity: 0.9 }).addTo(state.map);
  });
}

function renderStops() {
  state.stops.forEach((stop) => {
    const waiting = stop.queue_pax;
    const colour = stop.provenance?.kind === 'measured' ? '#4ea3f5' : '#6b7891';
    const icon = L.divIcon({
      html: `<div class="stop-dot${stop.id === state.stopId ? ' live' : ''}" style="background:${colour}"></div>`,
      className: '',
      iconSize: [15, 15],
      iconAnchor: [7, 7],
    });
    const marker = L.marker([stop.lat, stop.lng], { icon, title: stop.name }).addTo(state.map);
    marker.bindTooltip(
      `<b>${stop.name}</b><br>${waiting === null ? 'no measurement' : `${num(waiting)} waiting`}`,
      { direction: 'top', offset: [0, -10] },
    );
    marker.on('click', () => selectStop(stop.id));
    state.markers[stop.id] = marker;
  });
}

// ------------------------------------------------------------ stop switching

function selectStop(stopId) {
  if (!stopId) return;
  state.stopId = stopId;

  Object.entries(state.markers).forEach(([id, marker]) => {
    const dot = marker.getElement()?.querySelector('.stop-dot');
    if (dot) dot.classList.toggle('live', id === stopId);
  });

  const stop = state.stops.find((s) => s.id === stopId);
  if (!stop) return;

  $('stop-name').textContent = stop.name;
  $('stop-sub').textContent = stop.served_by_routes?.length
    ? `${stop.zone || ''} · routes ${stop.served_by_routes.join(', ')}`
    : `${stop.zone || ''} · no route serves this stop`;

  $('camera-note').textContent = stop.footage_provenance?.title
    ? `Camera: ${stop.footage_provenance.title}. ${stop.footage_provenance.disclosure || ''}`
    : 'No camera footage is mapped to this stop.';

  const video = $('video');
  video.src = stop.video_filename ? `/footage/${stop.video_filename}` : '';
  video.play().catch(() => {});

  applySnapshotFallback(stop);
  if (state.mode === 'live') startMonitoring();
  else stopMonitoring();
  refresh();
}

function applySnapshotFallback(stop) {
  const image = $('snapshot');
  const frames = stop.keyframes || {};
  if (frames.detections) {
    image.src = frames.detections;
    image.alt = 'Detections on a reference frame';
  } else {
    image.removeAttribute('src');
  }
  // The snapshot layer is only ever visible in Snapshots mode; leaving it loaded under
  // the live video made the pre-rendered banner bleed through the picture.
  const snapMode = state.mode === 'snap';
  image.hidden = !snapMode;
  $('video').hidden = snapMode;
}

// ------------------------------------------------------------------- modes

function setMode(mode) {
  state.mode = mode;
  document.body.classList.toggle('mode-live', mode === 'live');
  document.body.classList.toggle('mode-snap', mode === 'snap');
  $('mode-live').classList.toggle('active', mode === 'live');
  $('mode-snap').classList.toggle('active', mode === 'snap');
  $('mode-live').setAttribute('aria-selected', String(mode === 'live'));
  $('mode-snap').setAttribute('aria-selected', String(mode === 'snap'));

  $('video').hidden = mode !== 'live';
  // In live mode the canvas overlay draws the boxes; the pre-rendered snapshot JPEG
  // already has boxes and a provenance banner burned in, so it must not be visible.
  $('snapshot').hidden = mode !== 'snap';
  if (mode === 'snap') stopMonitoring();
  else startMonitoring();
}

// --------------------------------------------------------------- monitoring

async function startMonitoring() {
  if (state.monitoring) return;
  const stop = state.stops.find((s) => s.id === state.stopId);
  if (!stop?.video_filename) {
    setLiveBadge('off', 'no camera for this stop');
    return;
  }

  setLiveBadge('busy', 'starting monitor…');
  try {
    const result = await getJson(API.liveStart, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stop_id: state.stopId, fps: 5, segmentation: true, deep_pass_interval_s: 0 }),
    });
    state.monitoring = true;
    if (!result.zone_calibrated) {
      setLiveBadge('busy', 'monitoring (waiting zone not calibrated)');
    }
  } catch (error) {
    setLiveBadge('off', `cannot monitor: ${error.message}`);
  }
  refresh();
}

async function stopMonitoring() {
  if (!state.monitoring) return;
  state.monitoring = false;
  try {
    await getJson(API.liveStop, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stop_id: state.stopId }),
    });
  } catch (_) { /* stopping is best effort */ }
}

function setLiveBadge(kind, text) {
  const badge = $('live-badge');
  badge.className = `live-badge ${kind}`;
  $('live-text').textContent = text;
}

// -------------------------------------------------------------------- poll

function startPolling() {
  if (state.timers.stop) return;
  state.timers.stop = setInterval(refresh, 700);
}

async function refresh() {
  if (!state.stopId) return;
  let payload;
  try {
    payload = await getJson(API.live(state.stopId));
  } catch (_) {
    return;
  }
  if (payload.stop_id !== state.stopId) return;

  const live = payload.live;
  const recommendation = payload.recommendation || {};

  renderSentence(recommendation);
  renderNumbers(live, recommendation);
  renderFacts(live, payload);

  if (live) {
    state.overlay = live;
    drawOverlay();
    syncVideoToMonitor(live);
    if (state.mode === 'live' && state.monitoring) {
      const perf = live.performance || {};
      setLiveBadge('on', `monitoring · ${num(perf.detector_fps_actual, 1)} fps detection`);
    }
  } else if (state.monitoring) {
    setLiveBadge('busy', 'monitor starting…');
  }
}

/**
 * Keeps the playing video near the moment the monitor is reading.
 *
 * The boxes come from the monitor's frame position, so if the video element drifts
 * ahead the annotations lag the picture. Rather than hiding that, we nudge playback
 * back into alignment whenever the gap grows beyond a second.
 */
function syncVideoToMonitor(live) {
  const video = $('video');
  if (state.mode !== 'live' || !video || !video.duration) return;
  const target = live.video_time_s;
  if (target === null || target === undefined) return;
  const drift = video.currentTime - target;
  if (Math.abs(drift) > 1.0) {
    video.currentTime = target;
  }
}

function renderSentence(recommendation) {
  $('sentence').textContent = recommendation.sentence || '–';
}

function renderNumbers(live, recommendation) {
  const numbers = recommendation.numbers || {};
  $('waiting-count').textContent = num(currentWaiting(live, numbers));
  $('n-frame').textContent = live ? num(live.people_in_frame) : '–';
  $('n-arrivals').textContent = num(
    numbers.arrivals_per_min_displayed ?? numbers.arrivals_per_min,
    1,
  );
  $('n-headway').textContent = numbers.headway_recommended_min
    ? `${num(numbers.headway_recommended_min)} min`
    : '–';
}

/**
 * The displayed waiting count.
 *
 * These reference clips are seconds long and loop, so at the top of every lap the
 * tracker has to rebuild and the count collapses to zero. Rather than flash 0 several
 * times a minute, the last settled value is held briefly across a loop. It is never
 * held for long, and never when there is no monitor at all.
 */
const HOLD_AFTER_REBUILD_MS = 2500;

function currentWaiting(live, numbers) {
  if (!live) return numbers.waiting_passengers;

  const tracked = live.people_in_frame;
  const waiting = live.people_waiting;
  const now = Date.now();

  // A lap boundary is visible as: few people tracked, nobody yet classified as waiting,
  // and we previously had a settled reading.
  const rebuilding = tracked <= 2 || (waiting === 0 && state.lastWaiting > 0);
  if (waiting > 0) {
    state.lastWaiting = waiting;
    state.lastWaitingAt = now;
    return waiting;
  }
  if (rebuilding && state.lastWaiting > 0 && now - state.lastWaitingAt < HOLD_AFTER_REBUILD_MS) {
    return state.lastWaiting;
  }
  state.lastWaiting = waiting;
  return waiting;
}

function renderFacts(live, payload) {
  const facts = $('facts');
  facts.innerHTML = '';
  const add = (text, kind = '') => {
    const span = document.createElement('span');
    span.className = `fact ${kind}`;
    span.textContent = text;
    facts.appendChild(span);
  };

  const provenance = payload.provenance || {};
  if (provenance.kind === 'measured') {
    add(`arrival rate: ${provenance.confidence} confidence`,
      provenance.confidence === 'high' ? 'good' : provenance.confidence === 'medium' ? 'warn' : 'bad');
  } else {
    add('no measured arrival rate', 'bad');
  }

  if (live) {
    const seg = live.segmentation || {};
    const perf = live.performance || {};
    if (seg.enabled && seg.status === 'ok') {
      add(`${seg.model || 'segmentation'}: ${seg.masks} masks + ${seg.person_count ?? '–'} people, one pass`, 'good');
    } else if (seg.enabled) {
      add(`${seg.model || 'segmentation'}: ${seg.error || seg.status}`, 'warn');
    } else {
      add('segmentation: off', 'warn');
    }
    add(`boxes + masks on ${perf.detector_device || 'cpu'} · ${num(perf.detector_ms_per_frame, 0)} ms/frame`);
    const deep = live.deep_pass || {};
    if (deep.enabled) {
      add(`SAM 3 deep pass every ${num(deep.interval_s, 0)}s (${num(deep.last_duration_s, 1)}s each)`, 'warn');
    }
  }

  const source = payload.provenance?.clip_title;
  if (source) add(`footage: ${source}`);
}

// ------------------------------------------------------------------ canvas

function drawOverlay() {
  const live = state.overlay;
  const video = $('video');
  const canvas = $('overlay');
  const stage = $('stage');
  if (!live || !stage) return;

  const rect = stage.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);

  // Letterboxed video: work out where the picture actually sits inside the stage.
  const frame = live.frame_size || [16, 9];
  const scale = Math.min(rect.width / frame[0], rect.height / frame[1]);
  const drawW = frame[0] * scale;
  const drawH = frame[1] * scale;
  const offsetX = (rect.width - drawW) / 2;
  const offsetY = (rect.height - drawH) / 2;
  const px = (nx) => offsetX + nx * drawW;
  const py = (ny) => offsetY + ny * drawH;

  // Person masks underneath the boxes (same forward pass as the boxes).
  const seg = live.segmentation || {};
  const overlay = seg.live_overlay;
  if (overlay?.masks?.length) {
    ctx.fillStyle = 'rgba(214, 69, 139, 0.28)';
    ctx.strokeStyle = 'rgba(255, 200, 255, 0.55)';
    ctx.lineWidth = 1;
    overlay.masks.forEach((mask) => {
      (mask.polygons || []).forEach((polygon) => {
        if (polygon.length < 3) return;
        ctx.beginPath();
        polygon.forEach(([nx, ny], index) => {
          const x = px(nx);
          const y = py(ny);
          if (index === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      });
    });
  }

  // Waiting zone outline, so the counted region is visible.
  const bounds = live.waiting_zone?.bounds_frac;
  if (bounds) {
    ctx.strokeStyle = 'rgba(0, 200, 255, 0.65)';
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeRect(
      px(bounds.x1_frac), py(bounds.y1_frac),
      (bounds.x2_frac - bounds.x1_frac) * drawW,
      (bounds.y2_frac - bounds.y1_frac) * drawH,
    );
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(0, 200, 255, 0.9)';
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillText('waiting zone', px(bounds.x1_frac) + 4, py(bounds.y1_frac) + 13);
  }

  // Detection boxes: green when waiting, grey when just passing through.
  (live.tracks || []).forEach((track) => {
    const [nx1, ny1, nx2, ny2] = track.norm_box;
    const x = px(nx1);
    const y = py(ny1);
    const w = (nx2 - nx1) * drawW;
    const h = (ny2 - ny1) * drawH;

    ctx.lineWidth = track.waiting ? 2.5 : 1.5;
    ctx.strokeStyle = track.waiting ? 'rgba(82, 209, 139, 0.95)' : 'rgba(200, 210, 225, 0.55)';
    ctx.strokeRect(x, y, w, h);

    if (track.waiting && h > 30) {
      ctx.fillStyle = 'rgba(82, 209, 139, 0.95)';
      ctx.font = '600 10px Inter, sans-serif';
      ctx.fillText('waiting', x + 3, y - 4);
    }
  });

  $('frame-note').textContent =
    `${live.tracks?.length || 0} people tracked · ${live.people_in_waiting_zone} inside the zone · ` +
    `video ${num(live.video_time_s, 1)}s of ${num(live.clip_duration_s, 1)}s`;
}

// ---------------------------------------------------------------- scenarios

async function setScenario(key) {
  try {
    const payload = await getJson(API.scenario, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario: key }),
    });
    state.meta.current_scenario = payload.scenario;
    const stops = await getJson(API.stops);
    state.stops = stops.stops;
    refresh();
  } catch (error) {
    $('sentence').textContent = `Could not change demand setting: ${error.message}`;
  }
}

// -------------------------------------------------------------------- about

function buildAbout() {
  const body = $('about-body');
  const meta = state.meta;
  body.innerHTML = '';

  const p = (text) => body.appendChild(Object.assign(document.createElement('p'), { textContent: text }));
  const h = (text) => body.appendChild(Object.assign(document.createElement('h3'), { textContent: text }));
  const ul = (items) => {
    const list = document.createElement('ul');
    items.forEach((item) => list.appendChild(Object.assign(document.createElement('li'), { textContent: item })));
    body.appendChild(list);
  };

  p(meta.disclaimer);

  h('What the camera measures');
  ul([
    'People in view: detected by YOLOv8 in each frame, continuously while monitoring is on.',
    'People waiting: tracked over time and counted only if they are standing still inside the waiting area.',
    'Arriving per minute: counted from people actually entering the waiting area over the last minute of footage.',
  ]);

  h('How often each model runs');
  ul([
    'Detection: about 5 times a second, roughly 30-50 ms per frame on CPU (the live badge shows the rate actually achieved).',
    'Masks: a YOLO segmentation model draws one mask per person in the same forward pass as the boxes, about 40 ms per frame on CPU.',
    'SAM 3 is optional and much slower (about 14 seconds per frame here), so it is used only for one-off deep inspection, not for live monitoring.',
    'Headway advice: from the recorded analysis of this stop, not from the live counts, because a schedule decision needs minutes of data.',
  ]);

  h('Limits you should know');
  ul(meta.method.limitations);

  h('Where the footage comes from');
  p(meta.footage_note || 'Public reference footage from other cities, not RTL cameras.');
}

document.addEventListener('DOMContentLoaded', boot);
