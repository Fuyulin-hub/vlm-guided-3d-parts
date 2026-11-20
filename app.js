const buttons = [...document.querySelectorAll('[data-model]')];
const screen = document.querySelector('#load-screen');
const loadButton = document.querySelector('#load-model');
const title = document.querySelector('#load-title');
const status = document.querySelector('#load-status');
const rotate = document.querySelector('#rotate');
const reset = document.querySelector('#reset');
const download = document.querySelector('#download');
const partsList = document.querySelector('#parts-list');
const detailsPanel = document.querySelector('#selected-details');
const emptyPanel = document.querySelector('#inspector-empty');
const isolate = document.querySelector('#isolate');
const clearSelection = document.querySelector('#clear-selection');
const tooltip = document.querySelector('#part-tooltip');
const sizes = {pineapple:18,tractor:20,faucet:19,shovel:19};
let selected = 'pineapple', viewer, activeRequest = 0, initialized = false;
const modelData = fetch('assets/models.json?v=examples-2').then(response => {
  if (!response.ok) throw new Error('Model list unavailable');
  return response.json();
}).then(models => {
  for (const model of models) sizes[model.id] = Math.ceil(model.bytes / 1e6);
  if (!initialized) loadButton.textContent = `Load 3D model · ${sizes[selected]} MB`;
  return models;
});

function selectDetails(part) {
  detailsPanel.hidden = !part;
  emptyPanel.hidden = Boolean(part);
  isolate.disabled = clearSelection.disabled = !part;
  isolate.setAttribute('aria-pressed','false');
  isolate.textContent = 'Isolate part';
  rotate.setAttribute('aria-pressed','false');
  tooltip.hidden = true;
  for (const button of partsList.children) {
    const active = Number(button.dataset.partId) === part?.id;
    button.classList.toggle('selected',active);
    button.setAttribute('aria-pressed',String(active));
  }
  if (!part) return;
  document.querySelector('#part-name').textContent = part.name;
  document.querySelector('#part-name-source').textContent = part.nameSource;
  document.querySelector('#selected-swatch').style.background = `rgb(${part.color.join(',')})`;
  document.querySelector('#part-id').textContent = String(part.id).padStart(2,'0');
  document.querySelector('#part-area').textContent = `${part.surfacePercent.toFixed(2)}%`;
  document.querySelector('#part-faces').textContent = part.faceCount.toLocaleString('en-US');
  document.querySelector('#part-color').textContent = '#'+part.color.map(value=>value.toString(16).padStart(2,'0')).join('').toUpperCase();
}

function renderParts(details) {
  partsList.replaceChildren();
  document.querySelector('#parts-count').textContent = `${details.parts.length} regions`;
  document.querySelector('#inspector-message').textContent = 'Click the model or choose a part below.';
  for (const part of details.parts) {
    const button = document.createElement('button');
    button.className = 'part-row';
    button.dataset.partId = part.id;
    button.setAttribute('aria-pressed','false');
    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = `rgb(${part.color.join(',')})`;
    const name = document.createElement('span'); name.textContent = part.name;
    const share = document.createElement('span'); share.className = 'part-share';
    share.textContent = `${part.surfacePercent.toFixed(1)}%`;
    button.append(swatch,name,share);
    button.addEventListener('click',()=>viewer.selectPart(part.id));
    partsList.appendChild(button);
  }
  selectDetails(null);
}

function showHover(part,position) {
  tooltip.hidden = !part;
  if (!part) return;
  tooltip.textContent = `${part.name} · Region ${part.id}`;
  const rect = document.querySelector('.stage-shell').getBoundingClientRect();
  tooltip.style.left = `${Math.max(10,Math.min(position.x-rect.left+12,rect.width-180))}px`;
  tooltip.style.top = `${Math.max(45,position.y-rect.top-36)}px`;
}

function updateSelection(button) {
  selected = button.dataset.model;
  for (const b of buttons) { b.classList.toggle('active', b === button); b.setAttribute('aria-pressed', String(b === button)); }
  document.querySelector('#object-label').textContent = '/ ' + button.textContent.trim();
  download.href = `assets/${selected}.glb`;
  download.download = `anchorpart-${selected}.glb`;
  loadButton.textContent = `Load 3D model · ${sizes[selected]} MB`;
}

async function loadSelected() {
  const request = ++activeRequest;
  const model = selected;
  screen.hidden = false;
  title.textContent = 'Loading the 3D result…';
  status.textContent = 'Preparing the viewer.';
  loadButton.hidden = true;
  rotate.disabled = reset.disabled = true;
  partsList.replaceChildren(); selectDetails(null);
  document.querySelector('#parts-count').textContent = 'Loading…';
  document.querySelector('#inspector-message').textContent = 'Preparing the part details…';
  delete document.querySelector('#viewport').dataset.loaded;
  try {
    if (!viewer) {
      const module = await import('./parts-viewer.js?v=parts-1');
      if (request !== activeRequest) return;
      viewer = module.createViewer(document.querySelector('#viewport'), {onSelect:selectDetails,onHover:showHover});
    }
    initialized = true;
    const metadata = (await modelData).find(entry => entry.id === model);
    if (request !== activeRequest) return;
    const [detailsResponse,labelsResponse] = await Promise.all([
      fetch(`assets/${model}.parts.json`),fetch(`assets/${model}.face-ids.bin`)
    ]);
    if (!detailsResponse.ok || !labelsResponse.ok) throw new Error('Part details could not be loaded');
    const details = await detailsResponse.json();
    const bytes = await labelsResponse.arrayBuffer();
    const digest = await crypto.subtle.digest('SHA-256',bytes);
    const hash = Array.from(new Uint8Array(digest),v=>v.toString(16).padStart(2,'0')).join('');
    if (hash !== details.bindingSha256 || details.meshSha256 !== metadata.sha256) throw new Error('Part data does not match the selected model');
    if (request !== activeRequest) return;
    await viewer.load(`assets/${model}.glb`, metadata.viewRotation, details, new Uint16Array(bytes), (progress) => {
      if (request === activeRequest) status.textContent = progress > 0 ? `Loading ${Math.min(progress,100)}% · ${sizes[model]} MB` : 'Downloading the selected result…';
    });
    if (request !== activeRequest) return;
    renderParts(details);
    screen.hidden = true;
    rotate.disabled = reset.disabled = false;
    rotate.setAttribute('aria-pressed', 'false');
    document.querySelector('#viewport').dataset.loaded = model;
  } catch (error) {
    if (request !== activeRequest) return;
    title.textContent = 'The model could not load.';
    status.textContent = error.message.includes('WebGL') ? 'This browser needs WebGL to display 3D. You can still download the model or view the figures below.' : 'Check your connection and try again. You can also download the model.';
    loadButton.textContent = 'Try again';
    loadButton.hidden = false;
    document.querySelector('#parts-count').textContent = '—';
    document.querySelector('#inspector-message').textContent = 'Part details will appear after the model loads.';
    console.error(error);
  }
}

loadButton.addEventListener('click', loadSelected);
for (const button of buttons) button.addEventListener('click', () => {
  if (button.dataset.model === selected && initialized) return;
  updateSelection(button);
  if (initialized || loadButton.hidden) loadSelected();
});
rotate.addEventListener('click', () => {
  const on = rotate.getAttribute('aria-pressed') !== 'true';
  rotate.setAttribute('aria-pressed', String(on));
  viewer.setRotation(on);
});
reset.addEventListener('click', () => { viewer.reset(); rotate.setAttribute('aria-pressed', 'false'); });
clearSelection.addEventListener('click',()=>viewer.selectPart(null));
isolate.addEventListener('click',()=>{
  const on = isolate.getAttribute('aria-pressed') !== 'true';
  isolate.setAttribute('aria-pressed',String(on));
  isolate.textContent = on ? 'Show context' : 'Isolate part';
  viewer.isolate(on);
});
document.addEventListener('visibilitychange', () => viewer?.setPaused(document.hidden));
