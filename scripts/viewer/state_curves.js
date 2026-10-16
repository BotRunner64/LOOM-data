/* Contract-only state curves for the four exported state streams. */

const EEF_COMPONENTS = ['x', 'y', 'z', 'r1x', 'r1y', 'r1z', 'r2x', 'r2y', 'r2z'];
const PALETTE = [
  '#32b8ff', '#60d394', '#ffcb69', '#ff6b6b', '#b892ff',
  '#54e1d7', '#ff8fab', '#a7c957', '#f4a261', '#90a8ff',
];

export function buildCurveTraces(streams) {
  const traces = [];
  const definitions = [
    ['left_eef', 'Left EEF', EEF_COMPONENTS],
    ['right_eef', 'Right EEF', EEF_COMPONENTS],
    ['left_gripper', 'Left gripper', ['opening']],
    ['right_gripper', 'Right gripper', ['opening']],
  ];
  for (const [streamName, group, components] of definitions) {
    const stream = streams[streamName];
    if (!stream) continue;
    components.forEach((component, componentIndex) => {
      const values = streamName.endsWith('_eef')
        ? stream.values.map(value => value[componentIndex]) : stream.values;
      let min = Infinity, max = -Infinity;
      for (const value of values) {
        if (!Number.isFinite(value)) continue;
        min = Math.min(min, value); max = Math.max(max, value);
      }
      traces.push({
        id: `${streamName}:${component}`,
        streamName,
        group,
        component,
        times: stream.times,
        values,
        min: min === Infinity ? 0 : min,
        max: max === -Infinity ? 0 : max,
        // Keep the initial plot readable; rotation channels remain one click away.
        enabled: !streamName.endsWith('_eef') || componentIndex < 3,
        color: PALETTE[(componentIndex + (streamName.startsWith('right') ? 4 : 0)) % PALETTE.length],
      });
    });
  }
  return traces;
}

function niceValue(value) {
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(1);
  }
  return value.toFixed(Math.abs(value) < 1 ? 2 : 1);
}

export class StateCurves {
  constructor(canvas, controls, status, onSeek) {
    this.canvas = canvas;
    this.controls = controls;
    this.status = status;
    this.onSeek = onSeek;
    this.focusedStreamName = null;
    this.context = canvas.getContext('2d');
    canvas.tabIndex = 0;
    canvas.title = 'Click or drag to seek; use Left/Right arrows to step the focused state stream';
    this.currentNs = 0;
    this.dragging = false;
    this.onPointer = event => {
      if (!this.episode || (event.type === 'pointermove' && !this.dragging)) return;
      if (event.type === 'pointerdown') {
        if (event.button !== 0) return;
        canvas.focus({preventScroll: true});
        this.dragging = true;
        canvas.setPointerCapture(event.pointerId);
      }
      const bounds = canvas.getBoundingClientRect();
      const plotLeft = 52, plotRight = Math.max(plotLeft + 1, bounds.width - 14);
      const fraction = Math.max(0, Math.min(1, (event.clientX - bounds.left - plotLeft) /
        (plotRight - plotLeft)));
      onSeek(fraction * this.episode.end_ns);
    };
    this.onPointerUp = () => { this.dragging = false; };
    canvas.addEventListener('pointerdown', this.onPointer);
    canvas.addEventListener('pointermove', this.onPointer);
    canvas.addEventListener('pointerup', this.onPointerUp);
    canvas.addEventListener('pointercancel', this.onPointerUp);
    canvas.addEventListener('lostpointercapture', this.onPointerUp);
    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(canvas);
  }

  setEpisode(episode) {
    this.episode = episode;
    this.traces = buildCurveTraces(episode.streams);
    if (!episode.streams[this.focusedStreamName]) {
      this.focusedStreamName = this.traces[0]?.streamName ?? null;
    }
    this.renderControls();
    this.draw();
  }

  setCurrentTime(value) {
    this.currentNs = value;
    if (this.episode) {
      this.status.textContent = `${(value / 1e9).toFixed(2)} / ${(this.episode.end_ns / 1e9).toFixed(2)} s`;
    }
    this.draw();
  }

  renderControls() {
    this.controls.replaceChildren();
    const groups = new Map();
    for (const trace of this.traces) {
      if (!groups.has(trace.group)) groups.set(trace.group, []);
      groups.get(trace.group).push(trace);
    }
    for (const [name, traces] of groups) {
      const group = document.createElement('div'); group.className = 'curve-group';
      const heading = document.createElement('label'); heading.className = 'curve-group-title';
      const master = document.createElement('input'); master.type = 'checkbox';
      master.addEventListener('focus', () => { this.focusedStreamName = traces[0].streamName; });
      const updateMaster = () => {
        master.checked = traces.every(trace => trace.enabled);
        master.indeterminate = traces.some(trace => trace.enabled) && !master.checked;
      };
      heading.append(master, document.createTextNode(name));
      const options = document.createElement('div'); options.className = 'curve-options';
      for (const trace of traces) {
        const label = document.createElement('label'); label.className = 'curve-option';
        const input = document.createElement('input'); input.type = 'checkbox'; input.checked = trace.enabled;
        input.addEventListener('focus', () => { this.focusedStreamName = trace.streamName; });
        const swatch = document.createElement('span'); swatch.className = 'curve-swatch';
        swatch.style.backgroundColor = trace.color;
        input.addEventListener('change', () => { trace.enabled = input.checked; updateMaster(); this.draw(); });
        label.append(input, swatch, document.createTextNode(trace.component));
        options.append(label);
      }
      master.addEventListener('change', () => {
        traces.forEach(trace => { trace.enabled = master.checked; });
        options.querySelectorAll('input').forEach(input => { input.checked = master.checked; });
        updateMaster(); this.draw();
      });
      updateMaster(); group.append(heading, options); this.controls.append(group);
    }
  }

  stepStreamName() {
    return this.focusedStreamName;
  }

  draw() {
    const width = this.canvas.clientWidth, height = this.canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (this.canvas.width !== Math.round(width * ratio) || this.canvas.height !== Math.round(height * ratio)) {
      this.canvas.width = Math.round(width * ratio); this.canvas.height = Math.round(height * ratio);
    }
    const ctx = this.context;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const styles = getComputedStyle(this.canvas);
    const background = styles.getPropertyValue('--chart-bg').trim() || '#0c1520';
    const grid = styles.getPropertyValue('--chart-grid').trim() || '#26394d';
    const text = styles.getPropertyValue('--chart-text').trim() || '#9eafc2';
    ctx.fillStyle = background; ctx.fillRect(0, 0, width, height);
    if (!this.episode) return;

    const left = 52, right = width - 14, top = 16, bottom = height - 28;
    if (right <= left || bottom <= top) return;
    const enabled = this.traces.filter(trace => trace.enabled);
    let min = Math.min(...enabled.map(trace => trace.min));
    let max = Math.max(...enabled.map(trace => trace.max));
    if (!enabled.length) { min = -1; max = 1; }
    if (max === min) { const padding = Math.max(1, Math.abs(max) * 0.1); min -= padding; max += padding; }
    const padding = (max - min) * 0.08; min -= padding; max += padding;
    const x = ns => left + ns / Math.max(1, this.episode.end_ns) * (right - left);
    const y = value => bottom - (value - min) / (max - min) * (bottom - top);

    ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.fillStyle = text;
    ctx.font = '11px system-ui, sans-serif';
    for (let i = 0; i <= 4; i++) {
      const yy = top + (bottom - top) * i / 4;
      ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(niceValue(max - (max - min) * i / 4), left - 7, yy);
    }
    for (let i = 0; i <= 5; i++) {
      const xx = left + (right - left) * i / 5;
      ctx.beginPath(); ctx.moveTo(xx, top); ctx.lineTo(xx, bottom); ctx.stroke();
      ctx.textAlign = i === 0 ? 'left' : i === 5 ? 'right' : 'center';
      ctx.textBaseline = 'top';
      ctx.fillText((this.episode.end_ns / 1e9 * i / 5).toFixed(1) + 's', xx, bottom + 7);
    }

    ctx.save(); ctx.beginPath(); ctx.rect(left, top, right - left, bottom - top); ctx.clip();
    for (const trace of enabled) {
      ctx.beginPath(); ctx.strokeStyle = trace.color; ctx.lineWidth = trace.component === 'opening' ? 2.2 : 1.35;
      const stride = Math.max(1, Math.floor(trace.times.length / Math.max(1, (right - left) * 2)));
      let started = false;
      for (let index = 0; index < trace.times.length; index += stride) {
        const value = trace.values[index];
        if (!Number.isFinite(value)) { started = false; continue; }
        const xx = x(trace.times[index]), yy = y(value);
        if (started) ctx.lineTo(xx, yy); else { ctx.moveTo(xx, yy); started = true; }
      }
      const last = trace.times.length - 1;
      if (last >= 0 && Number.isFinite(trace.values[last])) ctx.lineTo(x(trace.times[last]), y(trace.values[last]));
      ctx.stroke();
    }
    const marker = x(this.currentNs);
    ctx.strokeStyle = '#ff8a4c'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(marker, top); ctx.lineTo(marker, bottom); ctx.stroke();
    ctx.restore();
  }
}
