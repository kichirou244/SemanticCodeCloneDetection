/**
 * Store tối giản.
 *
 * Nguyên tắc của Frontend Protocol: "càng ít global state càng tốt", đi từ
 * Local → Lifted → Global. Demo này chỉ có MỘT dữ liệu dùng chung cho cả trang
 * (cặp code đang xét + kết quả phân tích) nên một store ở cấp app là đủ, không
 * cần Zustand/Redux. Người gọi `subscribe` mới được vẽ lại.
 */

export function createStore(initialState) {
  let state = { ...initialState };
  const listeners = new Set();

  return {
    get: () => state,

    set(patch) {
      const next = typeof patch === 'function' ? patch(state) : patch;
      const merged = { ...state, ...next };
      const changed = Object.keys(next).filter((key) => merged[key] !== state[key]);
      if (changed.length === 0) return;
      state = merged;
      for (const listener of listeners) listener(state, changed);
    },

    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
