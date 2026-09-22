'use strict';
const $ = id => document.getElementById(id);
const radios = [...document.querySelectorAll('input[name=mode]')];
let health = null, selected = null, controller = null, urls = [];
const mode = () => radios.find(r => r.checked).value;
const decisions = {cv: 'Neural model', cv_no_ambiguity: 'Neural model; no OCR needed',
  ocr_override: 'Text verification changed the decision', cv_ocr_confirmed: 'Text verification confirmed',
  cv_ocr_unavailable: 'OCR unavailable; neural result', cv_ocr_inconclusive: 'OCR inconclusive; neural result',
  cv_ocr_error: 'OCR failed; neural result', blank_unchanged: 'Blank page; unchanged',
  uncertain_unchanged: 'Low confidence; unchanged', gemini: 'Gemini'};

function clearResult() {
  $('originalPreview').replaceChildren(); $('resultPreview').replaceChildren();
  urls.forEach(url => URL.revokeObjectURL(url)); urls = [];
  $('download').removeAttribute('href'); $('result').hidden = true;
  $('error').hidden = true; $('pages').replaceChildren();
}
function fail(message) { $('error').textContent = message; $('error').hidden = false; }
function update() {
  const selectedMode = mode(), available = health?.modes[selectedMode]?.available;
  $('remoteOptions').hidden = selectedMode !== 'genai';
  $('submit').disabled = !!controller || !selected || !available;
  $('file').disabled = !!controller;
  document.querySelectorAll('[data-sample]').forEach(b => { b.disabled = !!controller; });
  radios.forEach(r => { r.disabled = !!controller || health?.modes[r.value]?.available === false; });
  if (!health) return;
  const limit = selectedMode === 'genai' ? health.limits.genai_pdf_pages : health.limits.pdf_pages;
  $('limits').textContent = `Up to ${health.limits.file_mb} MiB · ${limit} PDF pages · single-frame images · quarter-turns only`;
  $('modeStatus').textContent = available
    ? selectedMode === 'genai' ? `Configured: ${health.modes.genai.model_version}. Provider access and quota are checked on each request.`
      : selectedMode === 'pure' ? 'Ready: EfficientNet-B0 at 384 × 384. Documents stay on this server.'
        : 'Ready: ResNet-18 at 224 × 224, with text verification for uncertain 0°/180° cases.'
    : 'This pipeline is unavailable. Check the server model or API-key configuration.';
}
function choose(file) {
  clearResult(); selected = file;
  $('selectedName').textContent = file?.name || '';
  $('status').textContent = '';
  if (file && health && file.size > health.limits.file_mb * 1024 * 1024) {
    fail(`File exceeds ${health.limits.file_mb} MiB.`); selected = null;
  }
  update();
}
function preview(target, blob, filename) {
  const url = URL.createObjectURL(blob); urls.push(url);
  if (filename.toLowerCase().endsWith('.pdf')) {
    const frame = document.createElement('iframe'); frame.src = url;
    frame.title = `${target === 'originalPreview' ? 'Original' : 'Corrected'} PDF preview`;
    $(target).append(frame);
  } else if (/\.tiff?$/i.test(filename)) {
    $(target).textContent = 'TIFF preview depends on your browser. Download the file to inspect it.';
  } else {
    const img = document.createElement('img'); img.src = url; img.alt = filename;
    img.onerror = () => { $(target).textContent = 'Preview unavailable in this browser. The download is still available.'; };
    $(target).append(img);
  }
}
radios.forEach(r => r.addEventListener('change', update));
$('file').addEventListener('change', () => choose($('file').files[0] || null));
document.querySelectorAll('[data-sample]').forEach(button => button.addEventListener('click', async () => {
  try {
    const response = await fetch(`/static/samples/${button.dataset.sample}`);
    if (!response.ok) throw new Error('Sample could not be loaded.');
    choose(new File([await response.blob()], button.dataset.sample, {type: response.headers.get('content-type')}));
    $('file').value = '';
  } catch (error) { fail(error.message); }
}));
$('cancel').addEventListener('click', () => controller?.abort());
$('form').addEventListener('submit', async event => {
  event.preventDefault(); if (!selected || controller) return;
  clearResult(); controller = new AbortController(); update();
  $('cancel').hidden = false; $('status').textContent = 'Processing your document…';
  const file = selected, selectedMode = mode(), started = performance.now();
  const timer = setTimeout(() => controller?.abort(), ((health?.limits.timeout_seconds || 45) + 25) * 1000);
  try {
    const form = new FormData(); form.append('file', file); form.append('mode', selectedMode);
    if (selectedMode === 'genai') form.append('genai_prompt', $('context').value);
    const response = await fetch('/correct-orientation', {method: 'POST', body: form, signal: controller.signal});
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const detail = typeof body.detail === 'string' ? body.detail : `Request failed (${response.status}).`;
      throw new Error(`${detail} Request: ${response.headers.get('x-request-id') || 'unknown'}`);
    }
    const output = await response.blob();
    const elapsed = ((performance.now() - started) / 1000).toFixed(2);
    const downloadUrl = URL.createObjectURL(output); urls.push(downloadUrl);
    $('download').href = downloadUrl; $('download').download = `corrected_${file.name}`;
    $('result').hidden = false;
    $('summary').textContent = `${selectedMode === 'pure' ? 'Pure CV' : selectedMode === 'hybrid' ? 'Hybrid' : 'Gemini'} · ${response.headers.get('x-page-count')} page(s) · ${elapsed}s request time`;
    $('reviewNotice').hidden = response.headers.get('x-needs-review') !== 'true';
    const results = JSON.parse(response.headers.get('x-page-results') || '[]');
    results.forEach(result => {
      const row = document.createElement('tr');
      [result.page, `${result.rotation_cw}°`, result.score === null ? 'Not provided' : result.score.toFixed(3),
        decisions[result.decision] || result.decision, result.needs_review ? 'Required' : 'Inspect result'].forEach(value => {
        const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
      }); $('pages').append(row);
    });
    $('requestId').textContent = `Request ID: ${response.headers.get('x-request-id')}`;
    preview('originalPreview', file, file.name); preview('resultPreview', output, file.name);
    $('status').textContent = 'Finished. Review the result before downloading.';
  } catch (error) {
    $('status').textContent = '';
    fail(error.name === 'AbortError' ? 'Request cancelled or timed out. The server may need a moment to finish its current work.' : error.message);
  } finally {
    clearTimeout(timer); controller = null; $('cancel').hidden = true; update();
  }
});
fetch('/health').then(async response => {
  if (!response.ok) throw new Error('Could not check the server. Reload and sign in if prompted.');
  health = await response.json();
  const initial = radios.find(r => r.value === health.inference_mode && health.modes[r.value]?.available)
    || radios.find(r => health.modes[r.value]?.available);
  if (initial) initial.checked = true;
  update();
}).catch(error => { $('modeStatus').textContent = error.message; });
window.addEventListener('beforeunload', () => { controller?.abort(); urls.forEach(url => URL.revokeObjectURL(url)); });
