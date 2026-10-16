// Keep the episode selector bounded even for manifests with hundreds of thousands of records.
export class EpisodeBrowser {
  constructor({select, previous, next, status}, onSelect) {
    Object.assign(this, {select, previous, next, status, onSelect});
    this.page = {episodes: [], offset: 0, limit: 100, total: 0};
    this.selectedId = null;
    this.generation = 0;
    this.setBusy(true);
    select.addEventListener('change', () => {
      const id = select.value;
      this.setSelected(id);
      onSelect(id);
    });
    previous.addEventListener('click', () => this.loadPage(Math.max(0, this.page.offset - this.page.limit)));
    next.addEventListener('click', () => this.loadPage(this.page.offset + this.page.limit));
  }

  setBusy(busy) {
    this.select.disabled = busy || !this.select.options.length;
    this.previous.disabled = busy || this.page.offset === 0;
    this.next.disabled = busy || this.page.offset + this.page.limit >= this.page.total;
    if (busy) this.status.textContent = 'Loading episodes…';
  }

  setSelected(id) {
    this.selectedId = id;
    this.render();
  }

  render() {
    const options = this.page.episodes.map(id => new Option(id, id));
    // An anomaly can jump to an episode outside the current page.
    if (this.selectedId && !this.page.episodes.includes(this.selectedId)) {
      options.unshift(new Option('Current: ' + this.selectedId, this.selectedId));
    }
    this.select.replaceChildren(...options);
    if (this.selectedId) this.select.value = this.selectedId;
  }

  async request(offset, initial = false) {
    this.controller?.abort();
    const controller = this.controller = new AbortController();
    const token = ++this.generation;
    this.setBusy(true);
    try {
      const query = new URLSearchParams({offset: String(offset), limit: '100'});
      const response = await fetch((initial ? './api/dataset?' : './api/episodes?') + query,
        {signal: controller.signal});
      if (!response.ok) throw new Error(await response.text());
      const page = await response.json();
      if (token !== this.generation) return null;
      this.page = page;
      if (initial) this.selectedId = page.episodes[0] ?? null;
      this.render();
      this.status.textContent = page.total
        ? `${page.offset + 1}–${page.offset + page.episodes.length} of ${page.total.toLocaleString()}`
        : 'No episodes';
      return page;
    } catch (error) {
      if (token !== this.generation || error.name === 'AbortError') return null;
      throw error;
    } finally {
      if (token === this.generation) this.setBusy(false);
    }
  }

  async initialize() {
    return await this.request(0, true);
  }

  async loadPage(offset) {
    try { await this.request(offset); }
    catch (error) { this.status.textContent = 'Failed to load episodes: ' + error.message; }
  }
}
