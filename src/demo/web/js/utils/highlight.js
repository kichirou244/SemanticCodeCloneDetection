/**
 * Vẽ token theo `kind` (cú pháp) và `b` (id khối khớp, -1 = lệch).
 *
 * Server đã quyết định hết thuật toán; hàm này chỉ dịch quyết định đó thành class.
 * Nhờ vậy frontend không cần biết `difflib`, không cần tự dò lại vị trí token, và
 * nếu đổi thuật toán căn chỉnh thì JS không phải sửa.
 */

/** Hue ấm cho phần lệch — đảo vòng 6 bậc để hai đoạn lệch liền nhau vẫn phân biệt được. */
const DIFF_HUES = [6, 22, 38, 350, 14, 32];

export function tokenClass(token) {
  const parts = ['tok', `tok--${token.k || 'ws'}`];
  if (token.b >= 0) parts.push('tok--match');
  else if (token.k !== 'ws') parts.push('tok--diff');
  return parts.join(' ');
}

export function tokenAttrs(token) {
  const attrs = { class: tokenClass(token) };
  if (token.b >= 0) {
    attrs.dataset = { block: token.b % 6 };
  } else {
    attrs.style = `--diff-diff-hue:${DIFF_HUES[hashCol(token) % DIFF_HUES.length]}`;
  }
  return attrs;
}

/**
 * Gán hue ấm cho token lệch một cách ổn định.
 * Dùng vị trí trong dòng thay vì hash nội dung: hai token lệch cạnh nhau phải khác
 * màu để mắt khoanh được từng đoạn, còn token giống nhau ở hai dòng khác thì không
 * cần khác.
 */
function hashCol(token) {
  let hash = 0;
  for (let i = 0; i < token.t.length; i += 1) {
    hash = (hash * 31 + token.t.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}
