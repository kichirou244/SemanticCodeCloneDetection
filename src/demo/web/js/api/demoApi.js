/** Gọi API. Một chỗ duy nhất biết đường dẫn và cách báo lỗi. */

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } catch {
      /* body không phải JSON — giữ nguyên detail gốc */
    }
    throw new Error(detail);
  }
  return response.json();
}

export const getStatus = () => request('/api/status');

export const getSamples = () => request('/api/samples');

export const getRealPairs = (split = 'test') =>
  request(`/api/samples/real?split=${encodeURIComponent(split)}`);

export const getRealSample = (pairId, split = 'test') =>
  request(`/api/samples/real/${encodeURIComponent(pairId)}?split=${encodeURIComponent(split)}`);

export const preload = (models) =>
  request('/api/preload', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ models }),
  });

export const analyze = (payload) =>
  request('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

export const getLibrary = () => request('/api/library');

export const getLibraryFile = (relPath) =>
  request(`/api/library/file?path=${encodeURIComponent(relPath)}`);

export const scanLibrary = (threshold) =>
  request(`/api/library/scan?threshold=${encodeURIComponent(threshold)}`, {
    method: 'POST',
  });

export const uploadZip = (file) => {
  const form = new FormData();
  form.append('file', file, file.name);
  return request('/api/uploads', { method: 'POST', body: form });
};

export const scanUpload = async (payload, onProgress) => {
  const resp = await fetch('/api/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`;
    try {
      const body = await resp.json();
      if (body && body.detail) detail = body.detail;
    } catch {
      /* body không phải JSON — giữ nguyên detail gốc */
    }
    throw new Error(detail);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let result = null;
  for (;;) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buf += decoder.decode(chunk.value, { stream: true });
    let idx;
    while ((idx = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      const msg = JSON.parse(line);
      if (msg.event === 'progress' && onProgress) onProgress(msg);
      else if (msg.event === 'result') result = msg.data;
      else if (msg.event === 'error') throw new Error(msg.message);
    }
  }
  if (!result) throw new Error('server ngắt giữa chừng, không có kết quả');
  return result;
};
