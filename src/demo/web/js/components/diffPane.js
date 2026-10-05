/**
 * Khung so khớp: hai khung code + cột nối giữa.
 *
 * Presentation-only. Nhận sẵn payload `alignment` từ server và dựng DOM.
 *
 * Về chiều cao dòng: `--code-line-h` cố định và `white-space: pre` (không xuống
 * dòng) là điều kiện để cột nối đặt được link theo `top = lineIndex * lineH`.
 * Nhãn đầu khung nằm ở grid row riêng (span cả 3 cột) chính vì lý do đó: nếu nhét
 * nó vào trong khung code thì cột nối phải cộng thêm chiều cao nhãn, và sai số
 * ngay khi font chưa tải xong.
 */

import { el, clear } from '../utils/dom.js';
import { tokenAttrs } from '../utils/highlight.js';
import { renderGutterLinks, bindBlockHover } from './gutter.js';

function renderLine(line, start) {
  const cells = line.tokens.length
    ? line.tokens.map((token, i) => {
        const s = el('span', tokenAttrs(token), token.t);
        // Chỉ số token phẳng (khớp `tok_a/tok_b` của evidence server gửi) —
        // placeholder dòng trống KHÔNG tốn chỉ số (nó không phải token payload).
        s.dataset.ti = start + i;
        return s;
      })
    : [el('span', { class: 'tok tok--ws', text: ' ' })];
  return el('div', { class: 'diff__line' }, [
    el('span', { class: 'diff__lineno', text: String(line.n) }),
    el('span', { class: 'diff__code' }, cells),
  ]);
}

function renderPane(side, modifier) {
  const pane = el('div', { class: `diff__pane diff__pane--${modifier}` });
  let ti = 0;
  for (const line of side.lines) {
    pane.append(renderLine(line, ti));
    ti += line.tokens.length;
  }
  return pane;
}

function headRow(labelA, labelB) {
  return el('div', { class: 'diff__head' }, [
    el('span', { class: 'diff__head-cell', text: `A · ${labelA}` }),
    el('span', { class: 'diff__head-cell diff__head-cell--gap' }),
    el('span', { class: 'diff__head-cell', text: `B · ${labelB}` }),
  ]);
}

/**
 * Dựng lại toàn bộ khung so khớp.
 * @param {HTMLElement} container
 * @param {object} alignment payload từ `AlignmentResult.to_payload()`
 * @param {{labelA: string, labelB: string}} labels
 */
export function renderDiff(container, alignment, labels) {
  clear(container);
  const grid = el('div', { class: 'diff__grid' }, [
    headRow(labels.labelA, labels.labelB),
    renderPane(alignment.a, 'a'),
    el('div', { class: 'diff__gutter', 'aria-hidden': 'true' }),
    renderPane(alignment.b, 'b'),
  ]);
  container.append(grid);
  renderGutterLinks(grid.querySelector('.diff__gutter'), alignment.blocks);
  bindBlockHover(grid);
  return grid;
}
