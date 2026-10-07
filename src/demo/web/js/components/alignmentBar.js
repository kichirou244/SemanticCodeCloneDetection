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
    ['Khớp', `${stats.n_blocks ?? '—'} khối · A ${covA !== undefined ? pct(covA) : '—'} · B ${covB !== undefined ? pct(covB) : '—'}`],
  ];
  for (const [label, value] of cells) {
    statsEl.append(el('span', {}, [el('b', { text: label }), `: ${value}`]), ' ');
  }
  if (differs) {
    statsEl.append(el('span', {
      class: 'is-warn',
      text: '· Mức khớp phủ < 60% — có thể không phải clone.',
    }));
  }
  // Hộp giải thích: chip # + sợi nối = khối giống nhau (rê chuột để khoanh,
  // không còn tô nền); vàng = vị trí model đang chọn chú ý tới.
  const note = el('div', { class: 'legend-note' });
  note.innerHTML =
    '<b>Cách đọc:</b><br>' +
    '<span class="sw sw--chip"></span><b>Chip # + sợi nối</b> = khối giống nhau, rê chuột để khoanh (không tô nền).<br>' +
    '<span class="sw sw--ev"></span><b>Vàng (gạch chân dày)</b> = chỗ model đang chọn chú ý tới — đổi model ở cột trái để xem model khác nhìn đâu.<br>' +
    '<span class="sw sw--plain"></span><b>Không màu</b> = phần còn lại, cứ đọc bình thường, không phải lỗi.';
  statsEl.append(note);

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