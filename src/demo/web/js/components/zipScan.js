/**
 * Quét file .zip bằng mọi model. Presentation + đọc lựa chọn từ DOM:
 *  - `readZipSelection` gom {models, threshold} cho main.js gửi đi.
 *  - `renderZipResults` vẽ bảng: mỗi hàng một cặp, mỗi model một ô điểm.
 *  - Click hàng → onPick({a, b}) để main.js nạp vào ô phân tích.
 *
 * Component này không gọi API (giống scanPanel.js): main.js làm việc đó.
 */

import { el, clear, fixed } from '../utils/dom.js';

/** Màu ô theo verdict — cùng ngữ nghĩa với scoreStrip. */
function cellClass(cell) {
  if (cell.score === null || cell.score === undefined) return 'zip-cell is-err';
  if (cell.verdict === 'clone') return 'zip-cell is-clone';
  if (cell.verdict === 'not_clone') return 'zip-cell is-not';
  return 'zip-cell';
}

function cellText(cell) {
  if (cell.score === null || cell.score === undefined) return 'lỗi';
  return fixed(cell.score, 3);
}

function cellTitle(modelId, cell) {
  if (cell.error) return `${modelId}: ${cell.error}`;
  if (cell.score === null || cell.score === undefined) return `${modelId}: không chấm được`;
  const thr = cell.threshold === null || cell.threshold === undefined ? '?' : cell.threshold;
  return `${modelId}: ${cell.score} (ngưỡng ${thr}, ${cell.verdict})`;
}

/** Đọc {models:[ids], threshold:number|null} từ DOM. */
export function readZipSelection({ modelsEl, sliderEl, frozenEl }) {
  const models = [...modelsEl.querySelectorAll('input[type=checkbox]:checked')]
    .map((c) => c.value);
  return { models, threshold: frozenEl.checked ? null : Number(sliderEl.value) / 100 };
}

export function renderZipModels(container, statuses) {
  clear(container);
  if (!statuses.length) {
    container.append(el('p', { class: 'scan-empty', text: 'Chưa biết models — đợi /api/status.' }));
    return;
  }
  for (const st of statuses) {
    const label = el('label', { class: 'zip-model' }, [
      el('input', { type: 'checkbox', value: st.model_id, checked: true }),
      el('span', { text: st.label || st.model_id }),
      el('small', { text: st.state }),
    ]);
    if (st.state === 'failed' && st.error) label.title = st.error;
    container.append(label);
  }
}

export function renderZipFiles(container, files, skipped) {
  clear(container);
  if (!files.length) return;
  const names = files.map((f) => `${f.name} (${f.chars} ký tự)`).join(', ');
  container.append(el('p', { class: 'scan-meta', text: `${files.length} file: ${names}` }));
  const skips = Object.entries(skipped || {}).filter(([, n]) => n > 0)
    .map(([k, n]) => `${k}: ${n}`).join(', ');
  if (skips) container.append(el('p', { class: 'scan-meta', text: `Bỏ qua — ${skips}.` }));
}

export function bindZipUpload({ inputEl, btnEl, onUpload, onEmpty }) {
  btnEl.addEventListener('click', () => {
    const file = inputEl.files && inputEl.files[0];
    if (!file) {
      onEmpty();
      inputEl.focus();
      return;
    }
    btnEl.disabled = true;
    onUpload(file).finally(() => { btnEl.disabled = false; });
  });
}

export function bindZipScan({ btnEl, sliderEl, valueEl, frozenEl, onScan }) {
  const paint = () => {
    valueEl.textContent = `${sliderEl.value}%`;
    sliderEl.disabled = frozenEl.checked;
  };
  sliderEl.addEventListener('input', paint);
  frozenEl.addEventListener('change', paint);
  paint();
  btnEl.addEventListener('click', () => {
    btnEl.disabled = true;
    onScan().finally(() => { btnEl.disabled = false; });
  });
}

function renderRow(pair, rank, modelIds, onPick) {
  const cells = modelIds.map((mid) => {
    const cell = (pair.scores || {})[mid] || { score: null };
    return el('span', { class: cellClass(cell), title: cellTitle(mid, cell), text: cellText(cell) });
  });
  const row = el('button', {
    type: 'button',
    class: 'scan-row',
    title: `Nạp cặp ${pair.a} ↔ ${pair.b} vào ô phân tích`,
  }, [
    el('span', { class: 'scan-row__rank', text: String(rank) }),
    el('span', { class: 'scan-row__pct', text: `${Math.round((pair.max_score || 0) * 100)}%` }),
    el('span', { class: 'scan-row__paths', text: `${pair.a} ↔ ${pair.b}` }),
    el('span', { class: 'scan-row__scores', text: '' }),
  ]);
  row.querySelector('.scan-row__scores').append(...cells);
  row.addEventListener('click', () => onPick(pair));
  return row;
}

/** Vẽ kết quả quét đa model. `modelIds` fix thứ tự cột theo lựa chọn lúc quét. */
export function renderZipResults(container, results, modelIds, onPick) {
  clear(container);
  if (!results || !results.pairs || !results.pairs.length) {
    container.append(el('p', {
      class: 'scan-empty',
      text: 'Không cặp nào đạt ngưỡng — hạ ngưỡng hoặc đổi models rồi quét lại.',
    }));
    return;
  }
  const head = el('div', { class: 'scan-row scan-row--head', text: '' }, [
    el('span', { class: 'scan-row__rank', text: '#' }),
    el('span', { class: 'scan-row__pct', text: 'max' }),
    el('span', { class: 'scan-row__paths', text: 'cặp file' }),
    el('span', { class: 'scan-row__scores', text: '' }),
  ]);
  head.querySelector('.scan-row__scores').append(
    ...modelIds.map((mid) => el('span', { class: 'zip-cell zip-cell--head', text: mid })),
  );
  container.append(head);
  results.pairs.forEach((pair, i) => container.append(renderRow(pair, i + 1, modelIds, onPick)));
  const ms = results.duration_ms;
  container.append(el('p', {
    class: 'scan-meta',
    text: `${results.n_kept}/${results.n_pairs} cặp đạt · ${ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`} · bấm hàng để nạp vào ô phân tích.`,
  }));
}

export function renderZipState(container, kind, message) {
  clear(container);
  container.append(el('p', {
    class: kind === 'error' ? 'scan-empty is-error' : 'scan-empty',
    text: message,
  }));
}
