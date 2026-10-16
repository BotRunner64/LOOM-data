import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';

const source = await readFile(new URL('../scripts/viewer/episodes.js', import.meta.url), 'utf8');
const {EpisodeBrowser} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));

// Minimal DOM boundary; requests can complete out of order even after cancellation.
class Element {
  value = '';
  options = [];
  events = {};
  addEventListener(name, callback) { this.events[name] = callback; }
  replaceChildren(...options) { this.options = options; this.value = options[0]?.value ?? ''; }
}
globalThis.Option = class {
  constructor(text, value) { Object.assign(this, {text, value}); }
};
const page = (offset = 0, total = 130849) => ({
  episodes: Array.from({length: Math.min(100, total - offset)}, (_, i) => `episode_${offset + i}`),
  offset, limit: 100, total, anomalies: null,
});
const response = value => ({ok: true, json: async () => value});
function createBrowser() {
  const elements = Object.fromEntries(['select', 'previous', 'next', 'status'].map(id => [id, new Element()]));
  const selected = [];
  const browser = new EpisodeBrowser(elements, id => selected.push(id));
  return {browser, ...elements, selected};
}

test('large datasets render one page and retain selection across pages and anomaly jumps', async () => {
  const requests = [];
  globalThis.fetch = async url => {
    requests.push(url);
    return response(page(Number(new URL(url, 'http://localhost').searchParams.get('offset'))));
  };
  const {browser, select, previous, next, selected} = createBrowser();
  const initial = await browser.initialize();
  assert.equal(initial.total, 130849);
  assert.equal(select.options.length, 100);
  assert.equal(select.value, 'episode_0');
  assert.equal(previous.disabled, true);
  assert.equal(next.disabled, false);
  await browser.loadPage(100);
  assert.equal(select.options.length, 101);
  assert.equal(select.value, 'episode_0');
  assert.deepEqual(selected, []);
  browser.setSelected('anomaly_episode_outside_page');
  assert.equal(select.options.length, 101);
  assert.equal(select.value, 'anomaly_episode_outside_page');
  assert.equal(select.options.some(option => option.value === 'episode_0'), false);
  select.value = 'episode_150';
  select.events.change();
  assert.equal(select.options.length, 100);
  assert.deepEqual(selected, ['episode_150']);
  assert.match(requests[0], /^\.\/api\/dataset\?/);
  assert.match(requests[1], /^\.\/api\/episodes\?/);
});

test('a stale response cannot replace a newer page', async () => {
  globalThis.fetch = async () => response(page());
  const {browser, select, status, next} = createBrowser();
  await browser.initialize();
  const pending = [];
  globalThis.fetch = (url, options) => new Promise(resolve => pending.push({url, options, resolve}));
  const older = browser.loadPage(100);
  const newer = browser.loadPage(200);
  assert.equal(pending[0].options.signal.aborted, true);
  pending[1].resolve(response(page(200, 201)));
  await newer;
  pending[0].resolve(response(page(100)));
  await older;
  assert.deepEqual(select.options.map(option => option.value), ['episode_0', 'episode_200']);
  assert.equal(status.textContent, '201–201 of 201');
  assert.equal(next.disabled, true);
});

test('empty pages and request failures keep the current episode available', async () => {
  globalThis.fetch = async () => response(page(0, 1));
  const {browser, select, previous, next, status} = createBrowser();
  await browser.initialize();
  globalThis.fetch = async () => response({episodes: [], offset: 0, limit: 100, total: 0});
  await browser.loadPage(0);
  assert.equal(select.value, 'episode_0');
  assert.equal(select.options.length, 1);
  assert.equal(status.textContent, 'No episodes');
  assert.equal(previous.disabled, true);
  assert.equal(next.disabled, true);
  globalThis.fetch = async () => ({ok: false, text: async () => 'server error'});
  await browser.loadPage(0);
  assert.equal(select.disabled, false);
  assert.equal(select.value, 'episode_0');
  assert.match(status.textContent, /server error/);
});

test('initial loading and pagination retain a notebook port-proxy prefix', async () => {
  const base = 'https://notebook.example/workspace/project/vscode/session/proxy/8765/';
  const requests = [];
  globalThis.fetch = async input => {
    const url = new URL(input, base);
    requests.push(url);
    // The notebook routes only URLs beneath the viewer's forwarded port.
    if (!url.pathname.startsWith(new URL(base).pathname)) {
      return {ok: false, text: async () => 'Notebook route not found'};
    }
    return response(page(Number(url.searchParams.get('offset'))));
  };
  const {browser, status} = createBrowser();
  await browser.initialize();
  await browser.loadPage(100);
  await browser.loadPage(0);
  assert.deepEqual(requests.map(url => url.pathname), [
    '/workspace/project/vscode/session/proxy/8765/api/dataset',
    '/workspace/project/vscode/session/proxy/8765/api/episodes',
    '/workspace/project/vscode/session/proxy/8765/api/episodes',
  ]);
  assert.equal(requests[2].searchParams.get('offset'), '0');
  assert.equal(status.textContent, '1–100 of 130,849');
});
