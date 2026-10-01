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
