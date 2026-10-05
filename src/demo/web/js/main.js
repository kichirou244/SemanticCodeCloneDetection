/**
 * Điểm khởi động trang.
 *
 * Luồng: nạp trạng thái model + danh sách mẫu → render thẻ điểm (skeleton nếu
 * model chưa nạp) → chờ người dùng bấm Phân tích → POST /api/analyze → render
 * thẻ điểm thật + khung so khớp + thanh thống kê.
 *
 * Quy tắc: mọi lỗi mạng/model hiển thị trên trang, không console-only. Một model
 * lỗi không làm hỏng các model khác — server đã đảm bảo điều đó, UI chỉ hiển thị.
 */

import { $, el } from './utils/dom.js';
import { getStatus, getSamples, getRealPairs, getRealSample, analyze, getLibrary, getLibraryFile, scanLibrary, uploadZip, scanUpload } from './api/demoApi.js';
import { createStore } from './state/store.js';
import { renderScoreStrip } from './components/scoreStrip.js';
import { renderDiff } from './components/diffPane.js';
import { mountSampleRail, mountRealRail } from './components/sampleRail.js';
import { mountAlignmentBar } from './components/alignmentBar.js';
import { bindScanControls, renderScan, setOnPickPair } from './components/scanPanel.js';
import { bindZipUpload, bindZipScan, readZipSelection, renderZipFiles, renderZipModels, renderZipResults, renderZipState } from './components/zipScan.js';

const state = createStore({
  status: null,
  samples: [],
  pairs: [],
  results: [],
  alignment: null,
  meta: null,
  library: null,
  scan: null,
  loading: false,
  error: null,
});

const refs = {
  btnAnalyze: $('#btn-analyze'),
  modeLiteral: $('#mode-literal'),
  modeAbstract: $('#mode-abstract'),
  inputA: $('#input-a'),
  inputB: $('#input-b'),
  sampleList: $('#sample-list'),
  sampleNote: $('#sample-note'),
  realList: $('#real-list'),
  scoreStrip: $('#score-strip'),
  diffView: $('#diff-view'),
  emptyState: $('#empty-state'),
  statsEl: $('#alignment-stats'),
  legendEl: $('#block-legend'),
  modeSwitch: $('#mode-switch'),
  scanPanel: $('#scan-panel'),
  scanResults: $('#scan-results'),
  scanNote: $('#scan-note'),
  scanSlider: $('#scan-threshold'),
  scanThresholdValue: $('#scan-threshold-value'),
  btnScan: $('#btn-scan'),
  zipFile: $('#zip-file'),
  btnUpload: $('#btn-upload'),
  zipModels: $('#zip-models'),
  zipThreshold: $('#zip-threshold'),
  zipThresholdValue: $('#zip-threshold-value'),
  zipFrozen: $('#zip-frozen'),
  btnZipScan: $('#btn-zip-scan'),
  zipResults: $('#zip-results'),
  zipNote: $('#zip-note'),
};

function setToast(message, kind = 'info') {
  const old = $('#toast');
  if (old) old.remove();
  const toast = el('div', { id: 'toast', class: `toast toast--${kind}` }, [
    el('span', { text: message }),
  ]);
  document.body.append(toast);
  requestAnimationFrame(() => toast.classList.add('is-visible'));
  setTimeout(() => toast.remove(), 4500);
}

const toastStyle = el('style', { text: `
  .toast {
    position: fixed; right: 1.25rem; bottom: 1.25rem; z-index: 60;
    max-width: 24rem; padding: 0.8rem 1rem;
    border-radius: var(--radius-md); background: var(--bg-raised);
    border: 1px solid var(--border-strong); box-shadow: var(--shadow-lg);
    font-size: 0.82rem; color: var(--text-primary);
    opacity: 0; transform: translateY(8px);
    transition: opacity var(--t-base), transform var(--t-base);
  }
  .toast.is-visible { opacity: 1; transform: none; }
  .toast--error { border-color: hsl(6 62% 56% / 55%); }
` });
document.head.append(toastStyle);

function currentMode() {
  return refs.modeAbstract.checked ? 'abstract' : 'literal';
}

function setBusy(busy) {
  state.set({ loading: busy });
  refs.btnAnalyze.disabled = busy;
  refs.btnAnalyze.textContent = busy ? 'Đang phân tích…' : 'Phân tích';
}

function renderScores() {
  renderScoreStrip(refs.scoreStrip, {
    statuses: state.get().status?.models ?? [],
    results: state.get().results,
    loading: state.get().loading,
  });
}

function renderEmpty() {
  $('.diff__grid', refs.diffView)?.remove();
  refs.diffView.append(refs.emptyState);
}

function renderResult(data) {
  const { alignment, models, meta } = data;
  state.set({ alignment, results: models, meta });

  if (alignment) {
    refs.emptyState.remove();
    const grid = renderDiff(refs.diffView, alignment, {
      labelA: `phía A · ${meta?.n_lines_a ?? '—'} dòng`,
      labelB: `phía B · ${meta?.n_lines_b ?? '—'} dòng`,
    });
    mountAlignmentBar({
      statsEl: refs.statsEl,
      legendEl: refs.legendEl,
      alignment,
      onBlockFocus: emitBlockFocus,
    });
  } else {
    renderEmpty();
  }
  renderScores();
}

function focusBlock(grid, blockId) {
  const tokens = grid.querySelectorAll('.tok');
  const links = grid.querySelectorAll('.gutter-link');
  for (const node of [...tokens, ...links]) {
    const b = Number(node.dataset.block ?? -1);
    node.classList.toggle('is-focus', blockId >= 0 && b === blockId);
    node.classList.toggle('is-dim', blockId >= 0 && !(b === blockId));
  }
}

function emitBlockFocus(blockId) {
  const grid = $('.diff__grid', refs.diffView);
  if (grid) focusBlock(grid, blockId);
}

async function runAnalyze() {
  const codeA = refs.inputA.value;
  const codeB = refs.inputB.value;
  if (!codeA.trim() || !codeB.trim()) {
    setToast('Điền cả hai ô code trước khi phân tích.', 'error');
    return;
  }
  setBusy(true);
  state.set({ error: null });
  try {
    const data = await analyze({
      code_a: codeA,
      code_b: codeB,
      mode: currentMode(),
    });
    renderResult(data);
  } catch (error) {
    setToast(`Phân tích thất bại: ${error.message}`, 'error');
    renderScores();
  } finally {
    setBusy(false);
  }
}

function fillFromSample(sample) {
  refs.inputA.value = sample.code_a;
  refs.inputB.value = sample.code_b;
  if (sample.mode === 'literal') refs.modeLiteral.checked = true;
  else if (sample.mode === 'abstract') refs.modeAbstract.checked = true;
}

async function pickSample(sample) {
  fillFromSample(sample);
  setToast(`Đã nạp mẫu “${sample.title}” — bắt đầu chấm điểm.`);
  await runAnalyze();
}

async function pickRealPair(pair) {
  setToast('Đang lấy cặp thật từ POJ-104…');
  refs.btnAnalyze.disabled = true;
  try {
    const data = await getRealSample(pair.pair_id, pair.split || 'test');
    fillFromSample(data);
    await runAnalyze();
  } catch (error) {
    setToast(`Không lấy được cặp thật: ${error.message}`, 'error');
  } finally {
    refs.btnAnalyze.disabled = false;
  }
}

/* ------------------------------------------------- quét zip đa model */

const zipState = { uploadId: null, filesByName: {} };

async function onZipUpload(file) {
  renderZipState(refs.zipResults, null, 'Đang tải zip lên…');
  try {
    const up = await uploadZip(file);
    zipState.uploadId = up.upload_id;
    zipState.filesByName = Object.fromEntries(up.files.map((f) => [f.name, f.code]));
    renderZipFiles(refs.zipNote, up.files, up.skipped);
    renderZipState(refs.zipResults, null,
      `Đã nạp ${up.files.length} file — chọn models rồi bấm Quét tất cả cặp.`);
  } catch (error) {
    renderZipState(refs.zipResults, 'error', error.message);
  }
}

async function onZipScan() {
  if (!zipState.uploadId) {
    renderZipState(refs.zipResults, 'error', 'Chưa tải file .zip lên.');
    return;
  }
  const sel = readZipSelection({
    modelsEl: refs.zipModels, sliderEl: refs.zipThreshold, frozenEl: refs.zipFrozen,
  });
  if (!sel.models.length) {
    renderZipState(refs.zipResults, 'error', 'Chọn ít nhất 1 model để chấm.');
    return;
  }
  renderZipState(refs.zipResults, null,
    'Đang chấm tất cả các cặp (M1 chậm nhất, vài phút với ~200 cặp)…');
  try {
    const results = await scanUpload({
      upload_id: zipState.uploadId, models: sel.models, threshold: sel.threshold,
    });
    state.set({ zipScan: results });
    renderZipResults(refs.zipResults, results, sel.models, pickZipPair);
  } catch (error) {
    renderZipState(refs.zipResults, 'error', error.message);
  }
}

async function pickZipPair(pair) {
  const a = zipState.filesByName[pair.a];
  const b = zipState.filesByName[pair.b];
  if (a === undefined || b === undefined) {
    setToast('Không còn nội dung file trong trang — tải zip lại.', 'error');
    return;
  }
  refs.inputA.value = a;
  refs.inputB.value = b;
  setToast(`Đã nạp cặp ${pair.a} ↔ ${pair.b} — bắt đầu chấm điểm.`);
  await runAnalyze();
}

function initZipScan() {
  const statuses = state.get().status?.models ?? [];
  renderZipModels(refs.zipModels, statuses);
  bindZipUpload({ inputEl: refs.zipFile, btnEl: refs.btnUpload, onUpload: onZipUpload });
  bindZipScan({
    btnEl: refs.btnZipScan, sliderEl: refs.zipThreshold,
    valueEl: refs.zipThresholdValue, frozenEl: refs.zipFrozen, onScan: onZipScan,
  });
  renderZipState(refs.zipResults, null, 'Chọn file .zip rồi Tải lên trước.');
}

/* ---------------------------------------------------------------- quét database */

function renderScanNote(lib) {
  refs.scanNote.textContent = lib.directory
    ? `Thư mục: ${lib.directory} · ${lib.n_files} file (quét tối đa ${lib.max_files})`
    : '';
}

async function runScan(threshold) {
  renderScan({ container: refs.scanResults, scanning: true });
  try {
    const results = await scanLibrary(threshold);
    state.set({ scan: results });
    renderScan({ container: refs.scanResults, results, onPickPair: pickScanPair });
  } catch (error) {
    renderScan({ container: refs.scanResults, error: error.message });
  } finally {
    refs.btnScan.disabled = false;
  }
}

async function pickScanPair(pair) {
  setToast(`Đang nạp cặp ${pair.a} ↔ ${pair.b}…`);
  try {
    const [fileA, fileB] = await Promise.all([
      getLibraryFile(pair.a),
      getLibraryFile(pair.b),
    ]);
    refs.inputA.value = fileA.code;
    refs.inputB.value = fileB.code;
    if (fileA.truncated || fileB.truncated) {
      setToast('File dài bị cắt theo giới hạn database khi tính điểm.', 'info');
    }
    await runAnalyze();
  } catch (error) {
    setToast(`Không đọc được file trong database: ${error.message}`, 'error');
  }
}

async function initLibrary() {
  try {
    const lib = await getLibrary();
    state.set({ library: lib });
    if (!lib.enabled) return; // chưa khai data.library_dir → giữ nguyên ẩn
    refs.scanPanel.hidden = false;
    refs.scanSlider.value = Math.round((lib.default_threshold ?? 0.5) * 100);
    renderScanNote(lib);
    renderScan({ container: refs.scanResults, files: lib.files ?? [] });
    bindScanControls({
      thresholdEl: refs.scanSlider,
      thresholdValueEl: refs.scanThresholdValue,
      btnScan: refs.btnScan,
      onScan: runScan,
    });
    setOnPickPair(pickScanPair);
  } catch (error) {
    setToast(`Không mở được database: ${error.message}`, 'error');
  }
}

/**
 * Lỗi /api/analyze trả 4xx hay payload có model lỗi — tất cả đều hiện được.
 * `ok=False` hiện thẻ đỏ kèm lý do, trang vẫn xem được phần alignment một cách độc lập.
 */
async function init() {
  mountSampleRail({
    listEl: refs.sampleList,
    noteEl: refs.sampleNote,
    samples: [],
    onPick: pickSample,
  });
  mountRealRail({ listEl: refs.realList, pairs: [], loading: true, onPick: pickRealPair });

  refs.btnAnalyze.addEventListener('click', runAnalyze);
  refs.modeSwitch.addEventListener('change', () => {
    if (state.get().alignment) runAnalyze();
  });
  renderScores();

  const [statusResult, samplesResult, pairsResult] = await Promise.allSettled([
    getStatus(), getSamples(), getRealPairs('test'),
  ]);

  if (statusResult.status === 'fulfilled') {
    state.set({ status: statusResult.value });
  } else {
    setToast(`Không liên hệ được server: ${statusResult.reason?.message ?? 'không rõ lý do'}`, 'error');
  }

  if (samplesResult.status === 'fulfilled') {
    // `/api/samples` trả list thuần; chấp cả hai hình dạng cho an toàn.
    const samples = Array.isArray(samplesResult.value)
      ? samplesResult.value
      : (samplesResult.value.samples ?? []);
    mountSampleRail({
      listEl: refs.sampleList,
      noteEl: refs.sampleNote,
      samples,
      onPick: pickSample,
    });
  } else {
    setToast(`Không nạp được danh sách mẫu: ${samplesResult.reason?.message ?? ''}`, 'error');
  }

  if (pairsResult.status === 'fulfilled') {
    const pairs = pairsResult.value.pairs ?? [];
    mountRealRail({ listEl: refs.realList, pairs, loading: false, onPick: pickRealPair });
  } else {
    mountRealRail({ listEl: refs.realList, pairs: [], loading: false, onPick: pickRealPair });
    if (refs.realList.firstChild) {
      refs.realList.firstChild.textContent = 'Không liệt kê được — xem log server.';
    }
  }

  initLibrary(); // nóng — không chặn UI, lỗi tự báo toast trong hàm
  initZipScan();
  renderScores();
}

init().catch((error) => {
  setToast(`Khởi động UI lỗi: ${error.message}`, 'error');
});