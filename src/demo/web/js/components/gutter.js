/**
 * Cột nối giữa hai khung + hiệu ứng rê chuột qua khối.
 *
 * Khối trong payload có `a:[lo,hi]` và `b:[lo,hi]` là khoảng DÒNG (1-indexed,
 * lo..hi-1). Link đặt theo công thức `top = (lineIndex - 1) * --code-line-h`
 * vì mỗi dòng có chiều cao cố định và không xuống dòng mềm (xem diffPane.js).
 */

import { el } from '../utils/dom.js';

const LINE_H = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--code-line-h')) || 24.8;

function linkTop(lineIndex1Based) {
  return (lineIndex1Based - 1) * LINE_H;
}

/** Dựng các "sợi nối" cho từng khối: một thanh ngang nối vùng A ↔ B. */
export function renderGutterLinks(gutter, blocks) {
  for (const block of blocks) {
    const top = Math.min(linkTop(block.a[0]), linkTop(block.b[0]));
    const bottom = Math.max(linkTop(block.a[1]), linkTop(block.b[1]));
    const mid = (top + bottom) / 2;
    // `data-block` dùng modulo 6 cho khớp với quy ước màu ở tokens.css — một bảng
    // `--block-hue` duy nhất cho cả token lẫn sợi nối (xem highlight.js).
    gutter.append(
      el('div', {
        class: 'gutter-link',
        dataset: { block: block.id % 6 },
        style: `top:${mid.toFixed(1)}px;`,
      }, [el('span', { class: 'gutter-link__label', text: `#${block.id}` })]),
    );
  }
}

/**
 * Rê chuột vào chip hay link của một khối → khoanh các token khối đó ở CẢ hai
 * khung, mờ phần còn lại. Unit là khối (id), client không cần biết vị trí.
 */
export function bindBlockHover(grid) {
  let active = -1;

  const apply = (blockId) => {
    active = blockId;
    grid.classList.toggle('is-hovering', blockId >= 0);
    const tokens = grid.querySelectorAll('.tok');
    for (const token of tokens) {
      const b = Number(token.dataset.block ?? -1);
      const focus = blockId >= 0 && b === blockId;
      token.classList.toggle('is-focus', focus);
      token.classList.toggle('is-dim', blockId >= 0 && !focus);
    }
    const links = grid.querySelectorAll('.gutter-link');
    for (const link of links) {
      const b = Number(link.dataset.block ?? -1);
      const focus = blockId >= 0 && b === blockId;
      link.classList.toggle('is-focus', focus);
      link.classList.toggle('is-dim', blockId >= 0 && !focus);
    }
  };

  grid.addEventListener('pointerover', (event) => {
    const blockId = Number(event.target.dataset.block ?? -1);
    if (blockId >= 0 && blockId !== active) apply(blockId);
  });
  grid.addEventListener('pointerout', (event) => {
    if (event.target.classList.contains('gutter-link') ||
        event.target.parentElement?.classList.contains('gutter-link')) return;
    apply(-1);
  });
  grid.addEventListener('pointerleave', () => apply(-1));
}

/** Dừng hiệu ứng (sau khi vẽ payload mới). */
export function clearBlockHover(grid) {
  grid.classList.remove('is-hovering');
  grid.querySelectorAll('.tok.is-focus, .tok.is-dim').forEach((node) => {
    node.classList.remove('is-focus', 'is-dim');
  });
  grid.querySelectorAll('.gutter-link.is-focus, .gutter-link.is-dim').forEach((node) => {
    node.classList.remove('is-focus', 'is-dim');
  });
}