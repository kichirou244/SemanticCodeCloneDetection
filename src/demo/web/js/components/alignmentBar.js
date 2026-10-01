/**
 * Thanh thống kê + chú giải khối ở đầu khung so khớp. Presentation-only.
 *
 * `stats` là payload từ `AlignmentResult.stats` (cover theo token, không theo dòng),
 * `blocks[n].id` quyết định chip nào được khoanh khi rê chuột.
 */

import { el, clear, pct } from '../utils/dom.js';

/**
 * Rê chuột vào chip → khoanh khối đó ở cả hai khung code.
 * Không chỉnh DOM ở đây: phát sự kiện để diffPane kích hoạt `is-focus`.
 */
export function mountAlignmentBar({ statsEl, legendEl, alignment, onBlockFocus }) {
  clear(statsEl);
  clear(legendEl);
  if (!alignment) return;

  const stats = alignment.stats || {};
  const covA = stats.coverage_a;
  const covB = stats.coverage_b;
  const differs = (covA !== undefined && covA < 0.6) || (covB !== undefined && covB < 0.6);

  const cells = [
    ['Khối khớp', stats.n_blocks ?? '—'],
    ['Token A khớp', covA !== undefined ? pct(covA) : '—'],
    ['Token B khớp', covB !== undefined ? pct(covB) : '—'],
    ['Đã khớp A/B',
      `${stats.n_matched_a ?? '—'} / ${stats.n_matched_b ?? '—'}`],
    ['Tổng A/B', `${stats.n_tokens_a ?? '—'} / ${stats.n_tokens_b ?? '—'}`],
  ];
  for (const [label, value] of cells) {
    statsEl.append(el('span', {}, [el('b', { text: label }), ` ${value}`]), ' ');
  }
  if (differs) {
    statsEl.append(el('span', {
      class: 'is-warn',
      text: '· Phần xanh phủ < 60% — có thể không phải clone.',
    }));
  }

  for (const block of alignment.blocks || []) {
    const chip = el('button', {
      type: 'button',
      class: 'block-chip',
      dataset: { block: block.id % 6 },
      title: `Khối #${block.id}: dòng ${block.a[0]}–${block.a[1] - 1} (A) ↔ ${block.b[0]}–${block.b[1] - 1} (B)`,
    }, [
      el('span', { class: 'block-chip__swatch' }),
      `#${block.id}`,
      el('span', { class: 'block-chip__link', text: `${block.a[0]}↔${block.b[0]}` }),
    ]);
    chip.addEventListener('pointerenter', () => onBlockFocus?.(block.id % 6));
    chip.addEventListener('pointerleave', () => onBlockFocus?.(-1));
    legendEl.append(chip);
  }
}