/**
 * Vẽ token theo `kind` (cú pháp) và `b` (id khối khớp, -1 = lệch).
 *
 * Server đã quyết định hết thuật toán; hàm này chỉ dịch quyết định đó thành class.
 * Nhờ vậy frontend không cần biết `difflib`, không cần tự dò lại vị trí token, và
 * nếu đổi thuật toán căn chỉnh thì JS không phải sửa.
 */

/**
 * Không còn tô mức "khớp" (bỏ màu xanh nền từ 07/10/2026): token khớp giữ
 * nguyên màu cú pháp. `dataset.block` vẫn được gán để rê chuột chip # / sợi
 * nối khoanh được khối. Màu tô duy nhất còn lại là vàng evidence (.tok-ev).
 */

export function tokenClass(token) {
  const parts = ['tok', `tok--${token.k || 'ws'}`];
  return parts.join(' ');
}

export function tokenAttrs(token) {
  const attrs = { class: tokenClass(token) };
  if (token.b >= 0) {
    attrs.dataset = { block: token.b % 6 };
  }
  // Token lệch: không gán màu riêng — giữ chữ cú pháp gốc, không nền.
  return attrs;
}

/* hashCol giữ lại để tương thích, hiện không dùng màu lệch nữa. */
function hashCol(token) {
  let hash = 0;
  for (let i = 0; i < token.t.length; i += 1) {
    hash = (hash * 31 + token.t.charCodeAt(i)) | 0;
  }
  return Math.abs(hash);
}
