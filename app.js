const buttons = [...document.querySelectorAll('[data-model]')];
const screen = document.querySelector('#load-screen');
const loadButton = document.querySelector('#load-model');
const title = document.querySelector('#load-title');
const status = document.querySelector('#load-status');
const rotate = document.querySelector('#rotate');
const reset = document.querySelector('#reset');
const download = document.querySelector('#download');
const sizes = {chair:19,mug:15,pineapple:18,ladder:17};
let selected = 'chair', viewer, activeRequest = 0, initialized = false;
const modelData = fetch('assets/models.json').then(response => {
  if (!response.ok) throw new Error('Model list unavailable');
  return response.json();
});

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
  try {
    if (!viewer) {
      const module = await import('./viewer.js');
      if (request !== activeRequest) return;
      viewer = module.createViewer(document.querySelector('#viewport'));
    }
    initialized = true;
    const metadata = (await modelData).find(entry => entry.id === model);
    if (request !== activeRequest) return;
    await viewer.load(`assets/${model}.glb`, metadata.viewRotation, (progress) => {
      if (request === activeRequest) status.textContent = progress > 0 ? `Loading ${Math.min(progress,100)}% · ${sizes[model]} MB` : 'Downloading the selected result…';
    });
    if (request !== activeRequest) return;
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
document.addEventListener('visibilitychange', () => viewer?.setPaused(document.hidden));
