/** Thẻ điểm của từng model. Presentation-only: nhận dữ liệu đã có, không gọi API. */

import { el, clear, fixed, pct } from '../utils/dom.js';

const VERDICT_TEXT = {
  clone: 'CLONE',
  not_clone: 'KHÔNG PHẢI',
  unknown: 'CHƯA CÓ NGƯỠNG',
};

/** Trạng thái nạp: hiện luôn cả "lần đầu sẽ chậm" để người xem không tưởng treo. */
const PENDING_STATES = { not_loaded: 'sẽ nạp lần đầu', loading: 'đang nạp…' };

function scoreCard(model) {
  const card = el('article', {
    class: `score-card score-card--${model.ok ? model.verdict : 'unknown'}`,
    dataset: { model: model.model_id },
  });

  card.append(
    el('div', { class: 'score-card__head' }, [
      el('span', { class: 'score-card__name', text: model.label, title: model.hint }),
      verdictBadge(model),
    ]),
  );

  if (!model.ok) {
    card.append(
      el('div', { class: 'score-card__error', text: model.error || 'không chấm được' }),
      metaLine(model),
    );
    return card;
  }

  card.append(
    el('div', { class: 'score-card__value', text: fixed(model.score, 4) }),
    scoreBar(model),
    metaLine(model),
  );
  return card;
}

function verdictBadge(model) {
  if (!model.ok) {
    return el('span', { class: 'verdict verdict--unknown', text: model.state || 'LỖI' });
  }
  const verdict = model.verdict || 'unknown';
  return el('span', {
    class: `verdict verdict--${verdict}`,
    text: VERDICT_TEXT[verdict] || verdict,
  });
}

/** Thanh điểm + vạch ngưỡng. Vạch là mốc tham chiếu trực quan nhất trên thẻ này. */
function scoreBar(model) {
  const width = Math.max(0, Math.min(1, model.score ?? 0)) * 100;
  const bar = el('div', { class: 'score-bar' }, [
    el('div', { class: 'score-bar__fill', style: `width:${width.toFixed(1)}%` }),
  ]);
  if (model.threshold !== null && model.threshold !== undefined) {
    bar.append(
      el('div', {
        class: 'score-bar__threshold',
        style: `left:${(model.threshold * 100).toFixed(1)}%`,
        dataset: { label: `ngưỡng ${model.threshold}` },
      }),
    );
  }
  return bar;
}

/** Số liệu phụ: độ trễ, F1 test thật của chính experiment đó, số token. */
function metaLine(model) {
  const parts = [];
  if (model.latency_ms !== null && model.latency_ms !== undefined) {
    parts.push(`${model.latency_ms} ms`);
  }
  if (model.threshold !== null && model.threshold !== undefined) {
    parts.push(`ngưỡng ${model.threshold}`);
  }
  const test = model.test_metrics || {};
  if (test.f1) parts.push(`F1 test ${pct(test.f1)}`);
  if (model.detail && model.detail.n_tokens) parts.push(`${model.detail.n_tokens} token`);
  if (model.detail && model.detail.truncated) parts.push('đã cắt 256 token');
  if (model.detail && model.detail.n_stmt_a !== undefined) {
    parts.push(`${model.detail.n_stmt_a}/${model.detail.n_stmt_b} stmt`);
  }
  return el('div', { class: 'score-card__foot' }, parts.map((text) => el('span', { text })));
}

/** Skeleton cho model chưa nạp — thay spinner đơn điệu (Frontend Protocol). */
function pendingCard(status) {
  const card = el('article', { class: 'score-card' }, [
    el('div', { class: 'score-card__head' }, [
      el('span', { class: 'score-card__name', text: status.label }),
    ]),
    el('div', { class: 'score-card__pending' }),
    el('div', { class: 'score-card__foot' }, [
      el('span', { text: PENDING_STATES[status.state] || status.state }),
    ]),
  ]);
  return card;
}

export function renderScoreStrip(container, { statuses = [], results = [], loading }) {
  clear(container);
  if (loading && results.length === 0) {
    for (const status of statuses) container.append(pendingCard(status));
    return;
  }
  // `test_metrics` chỉ có ở `/api/status`, không có trong `/api/analyze` — ghép lại
  // ở đây để thẻ điểm hiện được F1 test thật của chính experiment đó.
  const byId = new Map(statuses.map((status) => [status.model_id, status]));
  for (const result of results) {
    const status = byId.get(result.model_id);
    container.append(scoreCard({ ...result, test_metrics: status?.test_metrics }));
  }
}
