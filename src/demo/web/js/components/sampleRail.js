/**
 * Rail trái: ví dụ có sẵn + cặp thật POJ-104. Presentation-only + sự kiện.
 *
 * Có chủ đích chỉ phát sự kiện `pick(sample)` chứ không gọi API: rail không biết
 * thứ gọi là gì, main.js mới biết (nạp code + auto chạy phân tích).
 */

import { el, clear } from '../utils/dom.js';

const EXPECTED_META = {
  clone: ['Cùng ngữ nghĩa (thuộc "YES")'],
  not_clone: ['Khác ngữ nghĩa (thuộc "NO")'],
  ambiguous: ['Giữa hai lớp — mục tiêu để xem model chọn bên nào'],
};

function tagLabel(expected) {
  return { clone: 'CLONE', not_clone: 'KHÔNG PHẢI', ambiguous: 'MINH HOẠ' }[expected] ?? expected;
}

function renderSampleCard(sample, onPick) {
  const meta = [
    EXPECTED_META[sample.expected]?.[0] ?? `expected=${sample.expected}`,
    sample.mode ? `mode=${sample.mode}` : undefined,
  ].filter(Boolean).join(' · ');
  return el(
    'button', { type: 'button', class: 'sample-card', role: 'listitem' },
    [
      el('span', { class: 'sample-card__title' }, [
        el('span', { text: sample.title }),
        el('span', { class: `tag tag--${sample.expected}`, text: tagLabel(sample.expected) }),
      ]),
      el('span', { class: 'sample-card__meta', text: meta }),
    ],
  );
}

/** Gắn sự kiện: `onPick(sample)` gọi khi bấm mẫu có sẵn. */
export function mountSampleRail({ listEl, noteEl, samples = [], onPick }) {
  clear(listEl);
  if (noteEl) noteEl.textContent = '';
  for (const sample of samples) {
    const card = renderSampleCard(sample, onPick);
    card.addEventListener('click', () => onPick(sample));
    listEl.append(card);
  }
  const count = samples.length;
  if (noteEl) noteEl.textContent = `${count} mẫu tay kiểm tra cả 4 lỗi tên gọi điển hình.`;
}

function renderRealCard(pair, onPick) {
  const problemLine = pair.problem_a === pair.problem_b
    ? `cùng problem ${pair.problem_a}`
    : `problem ${pair.problem_a} ↔ ${pair.problem_b}`;
  const label = Number(pair.label) === 1 ? 'CLONE' : 'KHÔNG PHẢI';
  const meta = `${pair.split} · ${problemLine} · nhãn ${label}`;
  const title = `${pair.code_a_id} ↔ ${pair.code_b_id}`;
  return el('button', { type: 'button', class: 'real-card', role: 'listitem' }, [
    el('span', { class: 'real-card__title', title, text: pair.pair_id }),
    el('span', { class: 'real-card__meta', text: meta }),
  ]);
}

/** Gắn sự kiện: `onPick(pairId)` gọi khi bấm cặp thật. */
export function mountRealRail({ listEl, pairs = [], loading = false, onPick }) {
  clear(listEl);
  if (loading) {
    listEl.append(el('p', { class: 'hint', text: 'Đang liệt kê cặp test…' }));
    return;
  }
  if (pairs.length === 0) {
    listEl.append(el('p', { class: 'hint', text: 'Chưa thấy dữ liệu POJ-104 đã gom.' }));
    return;
  }
  for (const pair of pairs) {
    const card = renderRealCard(pair, onPick);
    card.addEventListener('click', () => onPick(pair));
    listEl.append(card);
  }
}