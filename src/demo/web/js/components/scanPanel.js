/**
 * Bảng quét thư mục database. Presentation-only + sự kiện:
 *  - `onScan(threshold)` khi bấm Quét.
 *  - `onPickPair(pair)` khi bấm một hàng → main.js nạp 2 file và chạy phân tích.
 *
 * Component này không gọi API: chỉ render dữ liệu do main.js đưa vào.
 */

import { el, clear } from '../utils/dom.js';

/** Nối tên 2 file thành một dòng hiển thị: `a.cpp ↔ b.c`. */
function pairLabel(pair) {
  return `${pair.a} ↔ ${pair.b}`;
}

function renderRow(pair, rank, onPick) {
  const similarity = Math.round((pair.similarity || 0) * 100);
  const row = el('button', {
    type: 'button',
    class: 'scan-row',
    title: `Xem so khớp ${pair.a} ↔ ${pair.b}`,
  }, [
    el('span', { class: 'scan-row__rank', text: String(rank) }),
    el('span', { class: 'scan-row__pct', text: `${similarity}%` }),
    el('span', { class: 'scan-row__paths', text: pairLabel(pair) }),
    el('span', { class: 'scan-row__fp', text: `${pair.n_fp_a}/${pair.n_fp_b} fp` }),
  ]);
  row.addEventListener('click', () => onPick(pair));
  return row;
}

/**
 * Gắn sự kiện vào các control một lần (gọi khi mount).
 * @param {object} refs các node DOM trong khối scan-panel
 */
export function bindScanControls({ thresholdEl, thresholdValueEl, btnScan, onScan }) {
  const renderLabel = () => {
    const value = Number(thresholdEl.value);
    thresholdValueEl.textContent = `${value}%`;
  };
  thresholdEl.addEventListener('input', renderLabel);
  renderLabel();
  btnScan.addEventListener('click', () => {
    const value = Number(thresholdEl.value) / 100;
    btnScan.disabled = true;
    onScan(value);
  });
}

/** Render trạng thái hiện tại của toàn bộ khối scan. */
export function renderScan({ container, files = [], scanning = false, results = null, error = null, onPickPair = () => {} }) {
  clear(container);
  if (error) {
    container.append(el('p', { class: 'scan-empty is-error', text: error }));
    return;
  }
  if (scanning) {
    container.append(el('p', { class: 'scan-empty', text: 'Đang fingerprint + so cặp… (all-vs-all N²)' }));
    return;
  }
  if (!results) {
    container.append(el('p', {
      class: 'scan-empty',
      text: `Database có ${files.length} file. Bấm Quét để tìm các cặp trên ngưỡng % — scan dùng MOSS (rất nhanh); bấm 1 hàng để xem chi tiết cả 4 model.`,
    }));
    return;
  }
  const pairs = results.pairs || [];
  if (pairs.length === 0) {
    container.append(el('p', {
      class: 'scan-empty',
      text: `Quét ${results.n_files} file xong sau ${results.duration_ms} ms — không có cặp nào đạt ngưỡng ${Math.round(results.threshold * 100)}%.`,
    }));
    return;
  }
  const list = el('div', { class: 'scan-list', role: 'list', 'aria-label': 'Các cặp trùng trên ngưỡng' });
  pairs.forEach((pair, index) => list.append(renderRow(pair, index + 1, onPickPair)));
  container.append(list);
}