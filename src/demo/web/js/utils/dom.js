/** Tiện ích DOM tối thiểu. Không dùng framework — payload nhỏ, không cần bundler. */

export const $ = (sel, root = document) => root.querySelector(sel);

/** Tạo phần tử với thuộc tính và con, trả về node. */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on')) node.addEventListener(key.slice(2).toLowerCase(), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(typeof child === 'string' ? document.createTextNode(child) : child);
  }
  return node;
}

/** Xoá sạch con của một container. */
export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** Định dạng phần trăm, luôn có 1 chữ số thập phân. */
export const pct = (value) => `${(value * 100).toFixed(1)}%`;

/** Định dạng số thập phân cố định, bỏ chữ số 0 thừa ở cuối. */
export function fixed(value, digits = 3) {
  if (value === null || value === undefined) return '—';
  return Number(value).toFixed(digits).replace(/\.?0+$/, '');
}
