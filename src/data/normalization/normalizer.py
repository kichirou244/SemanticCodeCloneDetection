"""CodeNormalizer — chuẩn hóa cho dedup/leakage, KHÔNG đổi semantics (Step 6).

Không rename identifier (đó là transformation RQ4, không phải preprocessing).
Raw code luôn được giữ nguyên ở `CodeSample.code`.
"""

import unicodedata


class CodeNormalizer:
    def normalize(self, code: str) -> str:
        text = unicodedata.normalize("NFC", code or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = [ln.rstrip() for ln in text.split("\n")]
        text = "\n".join(lines).strip() + "\n"
        return text
