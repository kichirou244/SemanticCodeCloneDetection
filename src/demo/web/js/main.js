/**
 * Điểm khởi động trang (chỉ luồng quét zip).
 *
 * Luồng duy nhất: chọn .zip → Tải lên → tick models → Quét tất cả cặp →
 * bảng kết quả (bấm hàng = xem highlight + điểm 5 model) + tải báo cáo
 * JSON/Markdown. Mọi lỗi hiện trên trang, không console-only.
 */

import { $, el } from './utils/dom.js';
import { getStatus, analyze, uploadZip, scanUpload } from './api/demoApi.js';
import { createStore } from './state/store.js';
import { renderScoreStrip } from './components/scoreStrip.js';
import { renderDiff } from './components/diffPane.js';
import { mountAlignmentBar } from './components/alignmentBar.js';
import {
  bindZipUpload, bindZipScan, readZipSelection, renderZipFiles,
  renderZipModels, renderZipResults, renderZipState,
  buildJsonReport, buildMarkdownReport,
} from './components/zipScan.js';

const state = createStore({
  status: null,
  results: [],
  alignment: null,
  meta: null,
  loading: false,
  error: null,
  zipScan: null,
});

const refs = {
  scoreStrip: $('#score-strip'),
  diffView: $('#diff-view'),
  emptyState: $('#empty-state'),
  statsEl: $('#alignment-stats'),
  legendEl: $('#block-legend'),
  zipFile: $('#zip-file'),
  zipFileName: $('#zip-file-name'),
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

function renderResult(data, labelA, labelB) {
  const { alignment, models, meta } = data;
  state.set({ alignment, results: models, meta });

  if (alignment) {
    refs.emptyState.remove();
    const tagA = labelA ? `${labelA} · ${meta?.n_lines_a ?? '—'} dòng` : `phía A · ${meta?.n_lines_a ?? '—'} dòng`;
    const tagB = labelB ? `${labelB} · ${meta?.n_lines_b ?? '—'} dòng` : `phía B · ${meta?.n_lines_b ?? '—'} dòng`;
    const grid = renderDiff(refs.diffView, alignment, { labelA: tagA, labelB: tagB });
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

async function runAnalyze(codeA, codeB, labelA, labelB) {
  setToast(`Đang chấm cặp ${labelA} ↔ ${labelB}…`);
  state.set({ error: null });
  try {
    const data = await analyze({ code_a: codeA, code_b: codeB, mode: 'abstract' });
    renderResult(data, labelA, labelB);
  } catch (error) {
    setToast(`Phân tích thất bại: ${error.message}`, 'error');
    renderScores();
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
  renderZipState(refs.zipResults, null, 'Bắt đầu chấm…');
  try {
    const results = await scanUpload({
      upload_id: zipState.uploadId, models: sel.models, threshold: sel.threshold,
    }, (prog) => renderZipState(refs.zipResults, null,
      `Đang chấm cặp ${prog.done}/${prog.total} (giữ lại ${prog.kept})…`));
    state.set({ zipScan: results });
    renderZipResults(refs.zipResults, results, sel.models, pickZipPair, onDownloadReport);
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
  await runAnalyze(a, b, pair.a, pair.b);
}

function downloadFile(name, text, type) {
  const blob = new Blob([text], { type });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

function onDownloadReport(format) {
  const results = state.get().zipScan;
  if (!results || !results.pairs) {
    setToast('Chưa có kết quả quét nào để tải.', 'error');
    return;
  }
  const files = Object.entries(zipState.filesByName)
    .map(([name, code]) => ({ name, chars: code.length, code }));
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  const payload = { files, results };
  if (format === 'json') {
    downloadFile(`ket-qua-quet-${stamp}.json`, buildJsonReport(payload), 'application/json');
  } else {
    downloadFile(`ket-qua-quet-${stamp}.md`, buildMarkdownReport(payload), 'text/markdown');
  }
  setToast(`Đã tải báo cáo ${format.toUpperCase()}.`);
}

function initZipScan() {
  const statuses = state.get().status?.models ?? [];
  renderZipModels(refs.zipModels, statuses);
  const noFile = () => renderZipState(
    refs.zipResults, 'error', 'Chưa chọn file .zip nào — bấm “Chọn .zip” trước.',
  );
  bindZipUpload({ inputEl: refs.zipFile, btnEl: refs.btnUpload, nameEl: refs.zipFileName, onUpload: onZipUpload, onEmpty: noFile });
  bindZipScan({
    btnEl: refs.btnZipScan, sliderEl: refs.zipThreshold,
    valueEl: refs.zipThresholdValue, frozenEl: refs.zipFrozen, onScan: onZipScan,
  });
  renderZipState(refs.zipResults, null, 'Chọn file .zip rồi Tải lên trước.');
}

/**
 * Lỗi /api/analyze trả 4xx hay payload có model lỗi — tất cả đều hiện được.
 * `ok=False` hiện thẻ đỏ kèm lý do, trang vẫn xem được phần alignment một cách độc lập.
 */
async function init() {
  renderScores();
  try {
    const status = await getStatus();
    state.set({ status });
  } catch (error) {
    setToast(`Không liên hệ được server: ${error.message ?? 'không rõ lý do'}`, 'error');
  }
  initZipScan();
  renderScores();
}

init();
