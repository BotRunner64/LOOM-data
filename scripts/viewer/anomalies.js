// This view consumes records only. Reason strings have no behavioral meaning.
export const STREAMS = ['left_eef', 'right_eef', 'left_gripper', 'right_gripper'];
export const containsTime = (record, ns) => record.start_ns === record.end_ns
  ? ns === record.start_ns : record.start_ns <= ns && ns < record.end_ns;
export function filterRecords(records, {episode = '', stream = '', reason = ''}) {
  return records.filter(record => (!episode || record.episode_id === episode) &&
    (!stream || record.stream === stream) && (!reason || record.reason === reason));
}
const decimalSeconds = ns => (ns / 1e9).toFixed(2);
const seconds = ns => decimalSeconds(ns) + ' s';
const interval = record => record.start_ns === record.end_ns ? seconds(record.start_ns)
  : `[${seconds(record.start_ns)}, ${seconds(record.end_ns)})`;
const label = record => `#${record.index} · ${record.episode_id} · ${record.stream} · ${record.reason} · ${interval(record)}`;
const streamNames = {left_eef: 'Left end effector', right_eef: 'Right end effector',
  left_gripper: 'Left gripper', right_gripper: 'Right gripper'};
const compactSeconds = decimalSeconds;

export class AnomalyReview {
  constructor(root, marks, onJump) {
    this.root = root;
    this.marks = marks;
    this.onJump = onJump;
    this.records = [];
    this.filtered = [];
    this.selected = null;
    this.episode = null;
    this.markButtons = [];
    this.get = id => root.querySelector('#' + id);
    this.panels = ['anomaly-list', 'anomaly-filters', 'anomaly-details'].map(this.get);
    for (const panel of this.panels) {
      panel.querySelector('summary').addEventListener('click', () => {
        for (const other of this.panels) if (other !== panel) other.open = false;
      });
    }
    document.addEventListener('click', event => {
      if (!root.contains(event.target)) this.closePanels();
    });
    root.addEventListener('keydown', event => {
      if (event.key !== 'Escape') return;
      const panel = this.panels.find(item => item.open);
      if (panel) { this.closePanels(); panel.querySelector('summary').focus(); }
    });
    for (const name of ['episode', 'stream', 'reason']) {
      this.get('anomaly-' + name).addEventListener('change', () => this.refresh());
    }
    this.get('anomaly-record').addEventListener('change', event => {
      if (event.target.value !== '') this.select(Number(event.target.value));
    });
    this.get('anomaly-prev').addEventListener('click', () => this.step(-1));
    this.get('anomaly-next').addEventListener('click', () => this.step(1));
    this.get('anomaly-reset').addEventListener('click', () => {
      for (const name of ['episode', 'stream', 'reason']) this.get('anomaly-' + name).value = '';
      this.refresh();
    });
  }
  closePanels() { for (const panel of this.panels) panel.open = false; }
  setReport(records) {
    this.root.hidden = records === null;
    this.records = (records || []).map((record, id) => ({...record, id}));
    for (const [name, field] of [['episode', 'episode_id'], ['stream', 'stream'], ['reason', 'reason']]) {
      const select = this.get('anomaly-' + name);
      select.replaceChildren(new Option('All ' + name + 's', ''));
      for (const value of new Set(this.records.map(record => record[field]))) {
        select.add(new Option(value, value));
      }
      select.disabled = !this.records.length;
    }
    this.refresh();
  }
  refresh() {
    this.filtered = filterRecords(this.records, Object.fromEntries(
      ['episode', 'stream', 'reason'].map(name => [name, this.get('anomaly-' + name).value])));
    if (!this.filtered.some(record => record.id === this.selected)) this.selected = null;
    const select = this.get('anomaly-record');
    select.replaceChildren(new Option(this.filtered.length ? 'Choose an anomaly…' : 'No anomalies', ''));
    for (const record of this.filtered) {
      const option = new Option(`#${record.index} · ${streamNames[record.stream]} · ${record.reason} · ${compactSeconds(record.start_ns)} s`, String(record.id));
      option.title = label(record);
      select.add(option);
    }
    select.disabled = !this.filtered.length;
    const filters = ['episode', 'stream', 'reason'].filter(name => this.get('anomaly-' + name).value);
    this.get('anomaly-filter-label').textContent = filters.length ? `Filters · ${filters.length}` : 'Filters';
    this.get('anomaly-filters').dataset.active = String(filters.length > 0);
    this.get('anomaly-reset').disabled = !filters.length;
    const scope = filters.map(name => `${name}: ${this.get('anomaly-' + name).value}`).join(' · ');
    this.get('anomaly-filter-label').title = scope || 'All episodes, streams and reasons';
    this.get('anomaly-list-caption').textContent = `${this.filtered.length} of ${this.records.length} records · ${scope || 'All episodes'}`;
    this.updateSelection();
    this.drawMarks();
  }
  updateSelection() {
    const index = this.filtered.findIndex(record => record.id === this.selected);
    this.get('anomaly-record').value = index < 0 ? '' : String(this.selected);
    this.get('anomaly-prev').disabled = index <= 0;
    this.get('anomaly-next').disabled = !this.filtered.length || index === this.filtered.length - 1;
    this.get('anomaly-count').textContent = `${index < 0 ? '–' : index + 1} / ${this.filtered.length}`;
    const record = this.selected === null ? null : this.records[this.selected];
    this.get('anomaly-title').textContent = record
      ? `${streamNames[record.stream]} · ${record.reason}`
      : !this.records.length ? 'No anomalies' : !this.filtered.length ? 'No matching anomalies' : 'Choose an anomaly to review';
    this.get('anomaly-timing').textContent = record
      ? record.start_ns === record.end_ns ? `${compactSeconds(record.start_ns)} s · Point`
        : `${compactSeconds(record.start_ns)}–${compactSeconds(record.end_ns)} s · Duration ${compactSeconds(record.end_ns - record.start_ns)} s`
      : !this.records.length ? 'The report is empty.' : !this.filtered.length ? 'Try clearing the filters.' : 'Use Next or open the record list.';
    this.get('anomaly-details').hidden = !record;
    if (!record) this.get('anomaly-details').open = false;
    this.get('anomaly-detail').textContent = record
      ? `Episode: ${record.episode_id}\nIndex: ${record.index}\nStream: ${record.stream}\nReason: ${record.reason}\nInterval: ${interval(record)}`
      : '';
    for (const [button, item] of this.markButtons) {
      button.setAttribute('aria-pressed', String(item.id === this.selected));
    }
  }
  select(id) {
    const record = this.records[id];
    if (!record) return;
    const listWasOpen = this.get('anomaly-list').open;
    this.closePanels();
    if (listWasOpen) this.get('anomaly-list').querySelector('summary').focus();
    this.selected = id;
    this.updateSelection();
    this.onJump(record);
  }
  step(direction) {
    const index = this.filtered.findIndex(record => record.id === this.selected);
    const record = this.filtered[index + direction];
    if (record) this.select(record.id);
  }
  setEpisode(episode) {
    this.episode = episode;
    if (this.selected !== null && this.records[this.selected].episode_id !== episode.episode_id) {
      this.selected = null;
    }
    this.drawMarks();
    this.updateSelection();
  }
  drawMarks() {
    this.marks.replaceChildren();
    this.markButtons = [];
    if (!this.episode) return;
    const records = this.filtered.filter(record => record.episode_id === this.episode.episode_id);
    for (const stream of STREAMS) {
      const items = records.filter(record => record.stream === stream);
      if (!items.length) continue;
      const row = document.createElement('div'); row.className = 'anomaly-lane';
      row.setAttribute('aria-label', stream + ' anomalies');
      const name = document.createElement('span'); name.textContent = stream;
      const track = document.createElement('div'); track.className = 'anomaly-track';
      for (const record of items) {
        const button = document.createElement('button'); button.type = 'button';
        button.className = 'anomaly-mark';
        const end = Math.max(1, this.episode.end_ns);
        button.style.left = Math.min(100, record.start_ns / end * 100) + '%';
        button.style.width = Math.max(0, (record.end_ns - record.start_ns) / end * 100) + '%';
        button.classList.toggle('point', record.start_ns === record.end_ns);
        button.title = label(record); button.setAttribute('aria-label', label(record));
        button.setAttribute('aria-pressed', String(record.id === this.selected));
        button.addEventListener('click', () => this.select(record.id));
        track.append(button); this.markButtons.push([button, record]);
      }
      row.append(name, track); this.marks.append(row);
    }
  }
  activeStreams(ns) {
    return new Set(this.records.filter(record => record.episode_id === this.episode?.episode_id &&
      containsTime(record, ns)).map(record => record.stream));
  }
}
