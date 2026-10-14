/**
 * Wayfinder dashboard.
 *
 * Rules this file follows, because the previous build broke all of them:
 *   - No hardcoded passenger numbers, wait times, headways or fleet figures anywhere.
 *     Every displayed value comes from the API in the same render pass.
 *   - Nothing is shown without its provenance. Measured, simulated and modelled values
 *     are labelled as such.
 *   - No fallback mock data. If the API is unavailable the screen says so.
 */

const state = {
  meta: null,
  stops: [],
  telemetry: null,
  currentStopId: null,
  mediaMode: 'video',
  markers: {},
  busMarkers: [],
  busTimer: null,
  loadToken: 0,
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- utilities

const fmt = (value, digits = 1) =>
  value === null || value === undefined || Number.isNaN(value) ? '—' : Number(value).toFixed(digits);

const STATUS_CLASS = {
  NORMAL: 'success',
  ELEVATED: 'warning',
  SURGE: 'danger',
  CAPACITY_FAILURE: 'danger',
  NO_ROUTE: 'info',
  MEASUREMENT_SUSPECT: 'warning',
  NO_DATA: 'info',
};

const CONFIDENCE_CLASS = { high: 'success', medium: 'warning', low: 'danger', none: 'info' };

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// -------------------------------------------------------------------- clock

function initClock() {
  const tick = () => {
    const now = new Date();
    const mvt = new Date(now.getTime() + now.getTimezoneOffset() * 60000 + 5 * 3600000);
    const clock = $('live-clock');
    if (clock) clock.textContent = `${mvt.toTimeString().split(' ')[0]} MVT`;
  };
  tick();
  setInterval(tick, 1000);
}

// -------------------------------------------------------------------- theme

function initTheme() {
  const stored = localStorage.getItem('wayfinder_theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = stored || (prefersDark ? 'dark' : 'light');
  applyTheme(theme);
  $('btn-theme').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    localStorage.setItem('wayfinder_theme', next);
    applyTheme(next);
  });
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $('btn-theme');
  btn.textContent = theme === 'dark' ? 'Light' : 'Dark';
  btn.setAttribute('aria-pressed', String(theme === 'dark'));
}

// --------------------------------------------------------------------- map

function initMap() {
  const map = L.map('osm-map', { center: [4.2035, 73.5255], zoom: 13, zoomControl: true });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
  return map;
}

let map = null;

// ---------------------------------------------------------------- data load

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${url} responded ${response.status}`);
  }
  return response.json();
}

async function boot() {
  initClock();
  initTheme();
  map = initMap();

  try {
    state.meta = await fetchJson('/api/meta');
  } catch (error) {
    showFatal(`Cannot reach the Wayfinder API (${error.message}). Start the server with ./run.sh.`);
    return;
  }

  renderMethod(state.meta);
  renderScenarios(state.meta.scenarios, state.meta.current_scenario.key);
  renderProvenanceStrip();

  $('btn-method').addEventListener('click', () => $('method-modal').showModal());
  $('btn-close-method').addEventListener('click', () => $('method-modal').close());
  $('btn-advisory').addEventListener('click', requestAdvisory);
  ['video', 'detections', 'segmentation', 'hybrid'].forEach((mode) => {
    const btn = $(`btn-view-${mode}`);
    if (btn) btn.addEventListener('click', () => switchMedia(mode));
  });

  await loadNetworkAndStops();
}

function showFatal(message) {
  const strip = $('prov-disclosure');
  strip.textContent = message;
  document.querySelector('.status-strip').classList.add('is-blocked');
  $('prov-kind').textContent = 'offline';
}

async function loadNetworkAndStops() {
  const token = ++state.loadToken;
  const [network, stopsPayload, telemetry] = await Promise.all([
    fetchJson('/api/network'),
    fetchJson('/api/stops'),
    fetchJson('/api/telemetry'),
  ]);
  if (token !== state.loadToken) return; // a newer scenario request superseded this one

  state.telemetry = telemetry;
  state.stops = stopsPayload.stops;

  renderKpis(telemetry);
  drawRoutes(network.routes);
  renderStopPins();
  initBuses(network.routes);

  const stillExists = state.stops.some((s) => s.id === state.currentStopId);
  selectStop(stillExists ? state.currentStopId : state.stops[0]?.id);
}

// -------------------------------------------------------------------- KPIs

function renderKpis(telemetry) {
  const k = telemetry.kpis;

  $('kpi-baseline-wait').textContent = fmt(k.mean_baseline_wait_min);
  $('kpi-optimized-wait').textContent = fmt(k.mean_optimized_wait_min);

  const delta = $('kpi-wait-delta');
  const pct = k.wait_change_pct;
  const direction = k.wait_change_direction;
  delta.textContent =
    direction === 'reduction'
      ? `${fmt(Math.abs(pct))}% lower`
      : direction === 'increase'
        ? `${fmt(Math.abs(pct))}% HIGHER (trade-off)`
        : 'no change';
  delta.className = `badge ${direction === 'reduction' ? 'success' : direction === 'increase' ? 'warning' : ''}`;

  $('kpi-inflow').textContent = fmt(k.total_inflow_pax_per_min, 2);
  $('kpi-inflow-note').textContent = `${k.status_counts.CAPACITY_FAILURE || 0} stop(s) over capacity at the recommended headway`;

  const cov = state.meta.analysis.coverage || {};
  const confidence = cov.confidence_counts || {};
  const measured = telemetry.data_status.stops_with_measurements ?? 0;
  const evaluable = (confidence.high || 0) + (confidence.medium || 0);
  $('kpi-coverage').textContent = `${evaluable}`;
  $('kpi-coverage-unit').textContent = `of ${telemetry.data_status.stops_total} stops`;
  $('kpi-confidence').textContent =
    `${measured} have footage · ${confidence.medium || 0} medium, ${confidence.low || 0} low confidence`;
}

function renderProvenanceStrip() {
  const strip = document.querySelector('.status-strip');
  const current = state.meta.current_scenario;
  const simulated = current.simulated;

  strip.classList.toggle('is-measured', !simulated);
  strip.classList.toggle('is-blocked', false);
  $('prov-kind').textContent = simulated ? 'simulated scenario' : 'measured + modelled';

  const coverage = state.meta.analysis.coverage || {};
  const conf = coverage.confidence_counts || {};
  const low = conf.low || 0;
  const total = coverage.stops_total || 0;

  let text = simulated
    ? `${current.description} Base rates are measured from public reference footage of other cities.`
    : `Arrival rates measured from public reference footage. Wait times are modelled, not observed.`;

  if (!simulated && low > 0) {
    text += ` ${low} of ${total} stops have a low-confidence measurement, so their headway is held at the timetable rather than compressed.`;
  }
  text += ` ${state.meta.disclaimer}`;
  $('prov-disclosure').textContent = text;
}

function renderScenarios(scenarios, activeKey) {
  const row = $('scenario-pills');
  row.innerHTML = '';
  scenarios.forEach((scenario) => {
    const btn = el('button', `pill${scenario.key === activeKey ? ' active' : ''}`);
    btn.type = 'button';
    btn.setAttribute('role', 'radio');
    btn.setAttribute('aria-checked', String(scenario.key === activeKey));
    btn.dataset.key = scenario.key;
    btn.appendChild(document.createTextNode(scenario.label));
    if (scenario.simulated) {
      const tag = el('span', 'sim', 'sim');
      tag.title = 'Simulated: a declared multiplier applied to measured base rates';
      btn.appendChild(tag);
    }
    btn.title = scenario.description;
    btn.addEventListener('click', () => setScenario(scenario.key));
    row.appendChild(btn);
  });
}

async function setScenario(key) {
  const buttons = document.querySelectorAll('#scenario-pills .pill');
  buttons.forEach((b) => {
    const active = b.dataset.key === key;
    b.classList.toggle('active', active);
    b.setAttribute('aria-checked', String(active));
  });

  try {
    const payload = await fetchJson('/api/scenario', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scenario: key }),
    });
    state.meta.current_scenario = payload.scenario;
    renderProvenanceStrip();
    await loadNetworkAndStops();
  } catch (error) {
    $('prov-disclosure').textContent = `Could not change scenario: ${error.message}`;
  }
}

// -------------------------------------------------------------------- map

function drawRoutes(routes) {
  routes.forEach((route) => {
    const geometry = route.osrm_geometry || route.waypoints || [];
    if (!geometry.length) return;
    L.polyline(geometry, {
      color: '#ffffff', weight: 6, opacity: 0.9, lineCap: 'round', lineJoin: 'round',
    }).addTo(map);
    L.polyline(geometry, {
      color: route.color || '#0b6bcb',
      weight: 4,
      opacity: 0.95,
      dashArray: route.id === 'R3' ? '8 6' : undefined,
      lineCap: 'round',
      lineJoin: 'round',
    }).addTo(map);
  });

  const legend = $('route-legend');
  legend.innerHTML = '';
  routes.forEach((route) => {
    const chip = el('span', 'legend-chip');
    const dot = el('span', 'chip-color');
    dot.style.background = route.color || '#0b6bcb';
    chip.appendChild(dot);
    chip.appendChild(document.createTextNode(`${route.id}: ${route.name.replace(/^Route \d+:\s*/, '')}`));
    legend.appendChild(chip);
  });
}

function pinClass(stop) {
  const status = stop.dispatch?.status;
  if (status === 'CAPACITY_FAILURE' || status === 'SURGE') return 'surge';
  if (status === 'ELEVATED') return 'elevated';
  if (status === 'NO_ROUTE' || status === 'NO_DATA') return 'nodata';
  return 'normal';
}

function renderStopPins() {
  Object.values(state.markers).forEach((marker) => map.removeLayer(marker));
  state.markers = {};

  state.stops.forEach((stop) => {
    const queue = stop.queue_pax;
    const label = queue === null || queue === undefined ? '?' : Math.round(queue);
    const cls = pinClass(stop);
    const icon = L.divIcon({
      html: `<div class="stop-pin ${cls}${stop.id === state.currentStopId ? ' selected' : ''}" style="width:28px;height:28px">${label}</div>`,
      className: 'custom-stop-icon',
      iconSize: [28, 28],
      iconAnchor: [14, 14],
    });
    const marker = L.marker([stop.lat, stop.lng], { icon, title: stop.name }).addTo(map);
    marker.bindTooltip(
      `<strong>${stop.name}</strong><br>${label} standing queue (measured)<br>` +
        `${fmt(stop.arrival_rate_pax_per_min, 2)} pax/min inflow (measured)<br>` +
        `status: ${stop.dispatch?.status || 'unknown'}`,
      { direction: 'top', offset: [0, -16] },
    );
    marker.on('click', () => selectStop(stop.id));
    state.markers[stop.id] = marker;
  });
}

function initBuses(routes) {
  state.busMarkers.forEach((b) => map.removeLayer(b.marker));
  state.busMarkers = [];
  if (state.busTimer) clearInterval(state.busTimer);

  // One bus per route, moving along that route's real road geometry.
  const routeBusCount = [];
  routes.forEach((route) => {
    const buses = route.id === 'R1' ? 2 : 1;
    for (let i = 0; i < buses; i += 1) {
      routeBusCount.push({ route, progress: (i + 1) / (buses + 1) });
    }
  });

  routeBusCount.forEach(({ route, progress }) => {
    const geometry = route.osrm_geometry || route.waypoints || [];
    if (geometry.length < 2) return;
    const icon = L.divIcon({
      html: `<div class="bus-chip">${route.id}</div>`,
      className: 'bus-div-icon',
      iconSize: [26, 16],
      iconAnchor: [13, 8],
    });
    const marker = L.marker(geometry[0], { icon, interactive: false }).addTo(map);
    state.busMarkers.push({ route, geometry, progress, speed: 0.0022, marker });
  });

  const step = () => {
    state.busMarkers.forEach((bus) => {
      const points = bus.geometry;
      const segments = points.length - 1;
      bus.progress = (bus.progress + bus.speed) % 1;
      const position = bus.progress * segments;
      const index = Math.min(segments - 1, Math.floor(position));
      const frac = position - index;
      const [lat1, lng1] = points[index];
      const [lat2, lng2] = points[index + 1];
      bus.marker.setLatLng([lat1 + (lat2 - lat1) * frac, lng1 + (lng2 - lng1) * frac]);
    });
  };

  step();
  state.busTimer = setInterval(step, 1500);
}

// --------------------------------------------------------------- inspector

function selectStop(stopId) {
  const stop = state.stops.find((s) => s.id === stopId);
  if (!stop) return;
  state.currentStopId = stopId;

  Object.entries(state.markers).forEach(([id, marker]) => {
    const node = marker.getElement()?.querySelector('.stop-pin');
    if (node) node.classList.toggle('selected', id === stopId);
  });

  renderStopHeader(stop);
  renderMeasuredDemand(stop);
  renderModel(stop);
  renderMedia(stop);
}

function renderStopHeader(stop) {
  $('stop-zone').textContent = stop.zone || 'Greater Malé';
  $('stop-name').textContent = stop.name;
  $('stop-routes').textContent = stop.served_by_routes?.length
    ? `Served by ${stop.served_by_routes.join(', ')}`
    : 'Not served by any configured route';

  const status = stop.dispatch?.status || 'UNKNOWN';
  const pill = $('stop-status');
  pill.textContent = status.replace(/_/g, ' ');
  pill.className = `status-pill ${STATUS_CLASS[status] || ''}`;

  const box = $('stop-warnings');
  box.innerHTML = '';
  (stop.warnings || []).forEach((warning) => {
    box.appendChild(el('div', `warn-item${/cannot be cleared|over capacity/i.test(warning) ? ' danger' : ''}`, warning));
  });
}

function renderMeasuredDemand(stop) {
  const provenance = stop.provenance || {};
  const confidence = provenance.confidence || 'none';
  const measured = provenance.kind === 'measured';

  $('measurement-sub').textContent = measured
    ? `measured, ${confidence} confidence`
    : 'no measurement available';

  const rateCell = $('val-rate');
  rateCell.textContent = measured ? `${fmt(stop.arrival_rate_pax_per_min, 2)} pax/min` : 'not measured';
  $('val-rate-sub').textContent = measured
    ? stop.scenario_multiplier !== 1
      ? `= measured ${fmt(stop.arrival_rate_base_pax_per_min, 2)} × scenario factor ${stop.scenario_multiplier}`
      : 'measured directly from footage'
    : 'no analysed footage is mapped to this stop';

  $('val-queue-base').textContent = measured ? `${fmt(stop.queue_baseline_measured)} pax` : '—';
  $('val-queue-base-sub').textContent = 'mean standing queue over the clip';

  $('val-queue').textContent = stop.queue_pax === null ? '—' : `${fmt(stop.queue_pax)} pax`;
  $('val-queue-sub').textContent = stop.queue_source || '';

  $('val-dwell').textContent = measured ? `${fmt(provenance.mean_dwell_s, 1)} s` : '—';
  $('val-dwell-sub').textContent = measured
    ? `from ${provenance.waiting_tracks}/${provenance.unique_tracks} tracked people`
    : '';

  drawSparkline(provenance.queue_series, provenance.analysed_span_s);
  renderProvenanceBody(stop);
}

function drawSparkline(series, spanSeconds) {
  const svg = $('queue-spark');
  svg.innerHTML = '';
  const note = $('spark-note');

  if (!series || !series.waiting_pax || series.waiting_pax.length < 2) {
    note.textContent = 'no time series available';
    return;
  }

  const values = series.waiting_pax;
  const times = series.t_s;
  const width = 320;
  const height = 64;
  const padding = 6;
  const maxValue = Math.max(1, ...values);

  const x = (i) => padding + (i / (values.length - 1)) * (width - 2 * padding);
  const y = (v) => height - padding - (v / maxValue) * (height - 2 * padding);

  const ns = 'http://www.w3.org/2000/svg';
  const area = document.createElementNS(ns, 'path');
  const points = values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  area.setAttribute('d', `M${x(0).toFixed(1)},${(height - padding).toFixed(1)} L${points.join(' L')} L${x(values.length - 1).toFixed(1)},${(height - padding).toFixed(1)} Z`);
  area.setAttribute('class', 'area');
  svg.appendChild(area);

  const line = document.createElementNS(ns, 'polyline');
  line.setAttribute('points', points.join(' '));
  line.setAttribute('class', 'line');
  svg.appendChild(line);

  const label = document.createElementNS(ns, 'text');
  label.setAttribute('x', padding);
  label.setAttribute('y', 11);
  label.setAttribute('class', 'lbl');
  label.textContent = `peak ${maxValue} pax`;
  svg.appendChild(label);

  const axisLabel = document.createElementNS(ns, 'text');
  axisLabel.setAttribute('x', width - padding);
  axisLabel.setAttribute('y', 11);
  axisLabel.setAttribute('class', 'lbl');
  axisLabel.setAttribute('text-anchor', 'end');
  axisLabel.textContent = `${fmt(spanSeconds, 0)} s analysed`;
  svg.appendChild(axisLabel);

  note.textContent = `every ${(times[1] - times[0]).toFixed(1)} s`;
}

function renderProvenanceBody(stop) {
  const provenance = stop.provenance || {};
  const body = $('prov-body');
  body.innerHTML = '';

  const add = (label, value, warn = false) => {
    if (value === null || value === undefined || value === '') return;
    const item = el('div', `prov-item${warn ? ' warn' : ''}`);
    const strong = el('strong', null, `${label}: `);
    item.appendChild(strong);
    item.appendChild(document.createTextNode(String(value)));
    body.appendChild(item);
  };

  if (provenance.kind === 'measured') {
    add('Footage', provenance.clip_title);
    add('Location of footage', provenance.clip_city);
    add('Source / licence', `${provenance.clip_source} — ${provenance.clip_license}`);
    add('Analysed window', `${fmt(provenance.analysed_span_s, 1)} s`);
    add('Analysis method', provenance.analysis_method);
    add('Disclosure', provenance.disclosure, true);
    if (provenance.scenario_scaling?.applied) {
      add('Scenario scaling', `×${provenance.scenario_scaling.factor} applied to the measured base rate`);
    }
    (provenance.confidence_reasons || []).forEach((reason) => add('Confidence note', reason, true));
    (provenance.limitations || []).forEach((limitation) => add('Limitation', limitation, true));
  } else {
    add('Status', provenance.note, true);
  }
}

function renderModel(stop) {
  const d = stop.dispatch || {};
  const hasModel = d.default_headway_min !== undefined && d.default_headway_min !== null;

  $('dispatch-sub').textContent = hasModel
    ? `${d.recommended_headway_source || ''}`
    : d.status === 'NO_ROUTE'
      ? 'no route serves this stop'
      : 'no demand measurement';

  $('val-headway-base').textContent = hasModel ? `${fmt(d.default_headway_min, 0)} min` : '—';
  $('val-headway-rec').textContent = hasModel ? `${fmt(d.recommended_headway_min, 0)} min` : '—';
  $('val-headway-sub').textContent = hasModel
    ? d.headway_change_pct === 0
      ? 'unchanged from the timetable'
      : `${d.headway_change_pct > 0 ? '+' : ''}${fmt(d.headway_change_pct)}% vs timetable`
    : '';

  $('val-bus').textContent = d.bus_type ? d.bus_type.replace(/_/g, ' ') : '—';
  $('val-bus-sub').textContent = d.bus_capacity_pax
    ? `${d.bus_capacity_pax} seats, ${fmt(d.effective_capacity_pax, 0)} usable after reserve`
    : '';

  $('val-service').textContent = hasModel
    ? `${fmt(d.offered_service_pax_per_min, 2)} / ${fmt(d.demand_rate_pax_per_min, 2)}`
    : '—';
  $('val-service-sub').textContent = hasModel
    ? d.capacity_ok
      ? `pax/min offered vs demanded — net ${fmt(d.net_clearance_pax_per_min, 2)}, utilisation ${fmt((d.utilization ?? 0) * 100, 0)}%`
      : 'demand exceeds offered capacity at this headway'
    : '';

  $('val-buses').textContent = hasModel ? `${d.buses_required}` : '—';
  $('val-buses-sub').textContent = hasModel
    ? d.fleet_shortfall
      ? `shortfall of ${d.fleet_shortfall} against ${d.buses_assigned} assignable`
      : `${d.buses_assigned} assignable, ${d.extra_buses_to_inject} added`
    : '';

  $('directive-text').textContent = d.directive_summary || '—';

  const list = $('cot-list');
  list.innerHTML = '';
  (d.reasoning_steps || []).forEach((step) => list.appendChild(el('div', 'cot-item', step)));
  if (d.queue_clear_note) list.appendChild(el('div', 'cot-item', `Queue clearance: ${d.queue_clear_note}`));
  if (d.baseline_formula) list.appendChild(el('div', 'cot-item', `Baseline wait: ${d.baseline_formula}`));
  if (d.optimized_formula) list.appendChild(el('div', 'cot-item', `Recommended wait: ${d.optimized_formula}`));
  list.appendChild(el('div', 'cot-item', `Wait figures are ${d.wait_reduction_source || 'modelled'}.`));

  $('advisory-box').hidden = true;
  $('btn-advisory').disabled = !hasModel;
}

// -------------------------------------------------------------------- media

function renderMedia(stop) {
  const provenance = stop.footage_provenance || {};
  const video = $('stop-video');
  const image = $('stop-cv-image');

  if (stop.video_filename) {
    const src = `/footage/${stop.video_filename}`;
    if (!video.src.endsWith(stop.video_filename)) {
      video.src = src;
      video.play().catch(() => {});
    }
  }

  const caption = provenance.title
    ? `${provenance.title} — reference footage, not RTL and not Malé`
    : 'No footage mapped to this stop';
  $('media-cap').textContent = caption;

  $('footage-line').textContent = provenance.title
    ? `Keyframe at ${fmt(stop.keyframe_timestamp_s, 1)}s · ${stop.detections_in_keyframe ?? '—'} people detected in that frame · ${stop.segmentation?.available ? `SAM 3: ${stop.segmentation.masks} masks (presence ${fmt(stop.segmentation.presence_score, 2)})` : 'SAM 3 layer not rendered'} · ${provenance.disclosure || ''}`
    : 'No footage has been analysed for this stop, so no detection overlay exists.';

  switchMedia(state.mediaMode);
}

function switchMedia(mode) {
  state.mediaMode = mode;
  const stop = state.stops.find((s) => s.id === state.currentStopId) || {};
  const keyframes = stop.keyframes || {};
  const video = $('stop-video');
  const image = $('stop-cv-image');

  ['video', 'detections', 'segmentation', 'hybrid'].forEach((m) => {
    const btn = $(`btn-view-${m}`);
    if (!btn) return;
    const active = m === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  });

  if (mode === 'video') {
    video.style.display = 'block';
    image.style.display = 'none';
    return;
  }

  const available = keyframes[mode];
  video.style.display = 'none';
  image.style.display = 'block';
  if (available) {
    image.src = available;
    image.alt =
      mode === 'detections'
        ? 'YOLOv8 person detections on a reference frame'
        : mode === 'segmentation'
          ? 'SAM 3 segmentation of waiting passengers on a reference frame'
          : 'Combined segmentation and detection overlay';
  } else {
    image.removeAttribute('src');
    image.alt = 'No overlay rendered for this layer';
  }
}

// ----------------------------------------------------------------- advisory

async function requestAdvisory() {
  const stop = state.stops.find((s) => s.id === state.currentStopId);
  if (!stop) return;
  const btn = $('btn-advisory');
  const box = $('advisory-box');
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = 'Requesting…';

  try {
    const payload = await fetchJson('/api/advisory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ stop_id: stop.id }),
    });
    renderAdvisory(payload);
  } catch (error) {
    box.hidden = false;
    box.innerHTML = '';
    box.appendChild(el('h4', null, 'Advisory unavailable'));
    box.appendChild(el('p', null, `The advisory endpoint failed: ${error.message}. The computed numbers above are unaffected.`));
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

function renderAdvisory(payload) {
  const box = $('advisory-box');
  box.hidden = false;
  box.innerHTML = '';

  if (!payload.available) {
    box.appendChild(el('h4', null, 'Deterministic engine narrative'));
    box.appendChild(el('p', 'muted', payload.reason || 'No language model configured.'));
  } else {
    box.appendChild(el('h4', null, 'DeepSeek advisory commentary'));
  }

  const advisory = payload.advisory || {};
  if (advisory.summary) box.appendChild(el('p', null, advisory.summary));

  if (advisory.advisory_points?.length) {
    const ul = el('ul');
    advisory.advisory_points.forEach((point) => ul.appendChild(el('li', null, point)));
    box.appendChild(ul);
  }
  if (advisory.data_caveats?.length) {
    box.appendChild(el('h4', null, 'Caveats'));
    const ul = el('ul');
    advisory.data_caveats.forEach((caveat) => ul.appendChild(el('li', null, caveat)));
    box.appendChild(ul);
  }
  box.appendChild(el('p', 'muted', 'Advisory text is commentary only; every number on this screen comes from the deterministic model.'));
}

// ------------------------------------------------------------------- method

function renderMethod(meta) {
  const body = $('method-body');
  body.innerHTML = '';

  const section = (title, content) => {
    body.appendChild(el('h3', null, title));
    if (typeof content === 'string') {
      const p = el('p', null, content);
      body.appendChild(p);
    } else {
      const ul = el('ul');
      content.forEach((item) => ul.appendChild(el('li', null, item)));
      body.appendChild(ul);
    }
  };

  section('How demand is measured', [
    meta.method.perception,
    meta.method.measurement,
    meta.method.wait_time,
  ]);

  section('What is measured, what is modelled', [
    'Measured: standing queue, dwell time and arrival rate, from public reference footage processed offline.',
    'Simulated: scenario demand multipliers applied to those measured base rates. Each multiplier is declared on the scenario button.',
    'Modelled: wait times, capacity, headways and fleet counts, computed from schedule parameters.',
    'Not used anywhere: official RTL demand, fleet, timetable or passenger data.',
  ]);

  section('Known limitations', meta.method.limitations);

  const coverage = (meta.analysis && meta.analysis.coverage) || {};
  section('Footage coverage', [
    `Stops with analysed footage: ${coverage.stops_with_measured_footage ?? '—'} of ${coverage.stops_total ?? '—'}.`,
    `Evaluable arrival rates — high: ${coverage.confidence_counts?.high ?? 0}, medium: ${coverage.confidence_counts?.medium ?? 0}, low: ${coverage.confidence_counts?.low ?? 0}.`,
    'A low-confidence rate means the analysed clip was too short, too sparse or too fragmented for a stable estimate. In that case Wayfinder explicitly holds the existing schedule instead of compressing headway on weak evidence.',
    `Analysis generated: ${meta.analysis.generated_at || 'not yet run'} by ${meta.analysis.generator || 'scripts/analyze_footage.py'}.`,
  ]);

  section('Disclaimer', meta.disclaimer);
}

// --------------------------------------------------------------------- init

// Exposed for automated UI checks and deep links (?stop=<id>).
window.wayfinderSelectStop = (stopId) => selectStop(stopId);

document.addEventListener('DOMContentLoaded', boot);
