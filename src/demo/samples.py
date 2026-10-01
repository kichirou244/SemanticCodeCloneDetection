"""Bộ code mẫu cho demo + đọc cặp thật từ POJ-104.

Hai nguồn dữ liệu, phục vụ hai mục đích khác nhau:

1. `BUILTIN_SAMPLES` — cặp viết tay, ngắn, được chọn để **minh hoạ từng loại
   clone** (Type-1/2/3) và cả trường hợp "vỏ giống, ngữ nghĩa khác". Không có
   nguồn nào tự nó chứa cặp đẹp bằng thứ được chọn có chủ đích.
2. Cặp thật từ `data/pairs/poj104/{split}.jsonl` — bằng chứng không phải ai
   cũng tự viết code ăn theo. Chỉ lấy theo `pair_id` mà người dùng bấm, không
   nhét sẵn vào response để khỏi phình.

Không gắn nhãn `expected` kiểu "model nên ra bao nhiêu" — nhãn chỉ là **quan điểm
nghiên cứu** ("đây là clone theo định nghĩa của bài báo"), còn điểm model cho ra
là dữ liệu. Trộn hai thứ đó lại chính là cách tự dệt dữ liệu.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

#: Mặc định giống hệt `configs/datasets/poj104.yaml`; demo không tự ý trỏ nơi khác.
DEFAULT_PAIRS_DIR = Path("data/pairs/poj104")
DEFAULT_SAMPLES_PATH = Path("data/processed/poj104/samples.jsonl")


@dataclass
class Sample:
    id: str
    title: str
    code_a: str
    code_b: str
    expected: str  # "clone" | "not_clone" | "ambiguous"
    note: str
    origin: str = "builtin"  # "builtin" | "poj104"


# --------------------------------------------------------------------- mẫu tay
BUILTIN_SAMPLES: list[Sample] = [
    Sample(
        id="type2-rename",
        title="Type-2 · Đổi tên biến và tham số",
        expected="clone",
        note="Cùng một thuật toán, chỉ khác tên `arr/total/i` và phép `+=` viết "
             "thành `= ... +`. Chế độ `literal` sẽ bôi đỏ hầu hết vì tên khác nhau; "
             "chế độ `abstract` bôi xanh gần hết. Đây là chỗ model semantic tỏa "
             "sáng và baseline từ khoá sụp.",
        code_a="""int sumArray(int arr[], int n) {
    int total = 0;
    for (int i = 0; i < n; i++) {
        total += arr[i];
    }
    return total;
}""",
        code_b="""int accumulate(int values[], int count) {
    int acc = 0;
    for (int k = 0; k < count; k++) {
        acc = acc + values[k];
    }
    return acc;
}""",
    ),
    Sample(
        id="type1-format",
        title="Type-1 · Đổi định dạng và thêm chú thích",
        expected="clone",
        note="Nội dung gốc không đổi một ký tự nào, chỉ khác kiểu xuống dòng, dấu "
             "`{}`, có thêm chú thích và khác tên hàm. Đây là loại clone dễ nhất — "
             "dùng để kiểm tra phần highlight có bám sát cú pháp hay không.",
        code_a="""int gcd(int a, int b) {
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a;
}""",
        code_b="""/* Euclid, ban do san bang thuong du nhat */
int greatestCommonDivisor(int x, int y)
{
    while (y != 0) {
        // chua so du, day len x
        int tmp = y; y = x % y; x = tmp;
    }
    return x;
}""",
    ),
    Sample(
        id="type3-restyle",
        title="Type-3 · Viết lại cấu trúc điều khiển",
        expected="clone",
        note="Cùng kết quả và cùng ý tưởng, nhưng thay `for` bằng `while`, đảo thứ tự "
             "so sánh, gộp phép cộng. Token khớp rải rác nên highlight sẽ chia thành "
             "nhiều khối màu khác nhau — nhìn thấy rõ điều này khó với model bag-of-token.",
        code_a="""int maxOf(int a, int b) {
    if (a > b) return a;
    return b;
}""",
        code_b="""int bigger(int p, int q) {
    int result = q;
    if (p > result) {
        result = p;
    }
    return result;
}""",
    ),
    Sample(
        id="near-miss-sort",
        title="Vỏ gần nhau · Ngữ nghĩa khác (bubble vs selection sort)",
        expected="ambiguous",
        note="Hai vòng lồng nhau, cùng biến `i/j`, cùng kiểu `int tmp` — nhưng một "
             "bên so sánh phần tử kề, bên kia tìm min. Đây là cặp mà **không nên** "
             "kỳ vọng mọi model trả lời giống nhau; dùng để soi điểm yếu thật.",
        code_a="""void bubbleSort(int a[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (a[j] > a[j + 1]) {
                int tmp = a[j];
                a[j] = a[j + 1];
                a[j + 1] = tmp;
            }
        }
    }
}""",
        code_b="""void selectionSort(int v[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int mn = i;
        for (int j = i + 1; j < n; j++) {
            if (v[j] < v[mn]) {
                mn = j;
            }
        }
        if (mn != i) {
            int tmp = v[i];
            v[i] = v[mn];
            v[mn] = tmp;
        }
    }
}""",
    ),
    Sample(
        id="unrelated",
        title="Không liên quan · Hai hàm khác hẳn",
        expected="not_clone",
        note="Hai hàm cùng kiểu trả về `int` và cùng có một vòng lặp, nhưng làm hai "
             "việc hoàn toàn khác. Dùng làm mốc: nếu bất kỳ model nào cho điểm cao "
             "ở đây thì đáng ngờ ngay, kể cả khi F1 trên POJ-104 rất đẹp.",
        code_a="""long factorial(int n) {
    long r = 1;
    for (int i = 2; i <= n; i++) {
        r *= i;
    }
    return r;
}""",
        code_b="""int countWords(const char *s) {
    int n = 0;
    int inWord = 0;
    for (int i = 0; s[i] != 0; i++) {
        if (s[i] == ' ') {
            inWord = 0;
        } else if (inWord == 0) {
            inWord = 1;
            n++;
        }
    }
    return n;
}""",
    ),
]


# ------------------------------------------------------------- cặp thật POJ-104
class CodeIndex:
    """Tra code theo `sample_id` mà không nạp hết 41 MB vào RAM.

    Lần đầu quét `samples.jsonl` một lần, chỉ ghi nhớ **byte offset** của từng
    dòng; sau đó `seek` thẳng tới offset để đọc. Quét một lần ~1-2 s, các lần sau
    tức thì — chấp nhận được cho demo, và tránh giữ 41 MB string trong RAM.
    """

    def __init__(self, samples_path: Path):
        self.samples_path = samples_path
        self._offsets: dict[str, int] | None = None

    def _build(self) -> dict[str, int]:
        if self._offsets is not None:
            return self._offsets
        offsets: dict[str, int] = {}
        with open(self.samples_path, "rb") as f:
            offset = 0
            for raw in f:
                stripped = raw.strip()
                if stripped:
                    sample_id = json.loads(stripped)["sample_id"]
                    offsets.setdefault(sample_id, offset)
                offset += len(raw)
        self._offsets = offsets
        return offsets

    def get(self, sample_id: str) -> str:
        offsets = self._build()
        if sample_id not in offsets:
            raise KeyError(f"không thấy sample_id {sample_id!r} trong {self.samples_path}")
        with open(self.samples_path, "rb") as f:
            f.seek(offsets[sample_id])
            return json.loads(f.readline())["code"]


def list_real_pairs(
    pairs_dir: Path = DEFAULT_PAIRS_DIR, split: str = "test", limit: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Chọn ngẫu nhiên nhưng tất định `limit` cặp thật từ một split.

    Tất định theo `seed` để mỗi lần bấm nút thấy cùng một danh sách — nếu không,
    người xem demo tưởng mình gặp lỗi không ổn định.
    """
    path = pairs_dir / f"{split}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(f"{seed}/{split}")
    rng.shuffle(pairs)
    picked = pairs[: max(0, limit)]
    return [
        {
            "pair_id": p["pair_id"],
            "split": split,
            "label": p["label"],
            "code_a_id": p["code_a_id"],
            "code_b_id": p["code_b_id"],
            "problem_a": p.get("problem_a", ""),
            "problem_b": p.get("problem_b", ""),
        }
        for p in picked
    ]


def load_real_pair(
    pair_id: str,
    index: CodeIndex,
    pairs_dir: Path = DEFAULT_PAIRS_DIR,
    split: str = "test",
) -> Sample:
    """Dựng một `Sample` từ `pair_id` thật, kèm code lấy từ `samples.jsonl`."""
    ref = _find_pair_ref(pairs_dir / f"{split}.jsonl", pair_id)
    return Sample(
        id=pair_id,
        title=f"POJ-104 {split} · {pair_id} · nhãn thật = {ref['label']}",
        code_a=index.get(ref["code_a_id"]),
        code_b=index.get(ref["code_b_id"]),
        expected="clone" if ref["label"] == 1 else "not_clone",
        note=("Nhãn lấy từ dataset: 1 = hai hàm cùng một problem (DEC-011), "
              "0 = khác problem. Đây là cặp mà chính EXP-00x đã dùng để tính "
              "F1 trên POJ-104."),
        origin="poj104",
    )


def _find_pair_ref(path: Path, pair_id: str) -> dict:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["pair_id"] == pair_id:
                return rec
    raise KeyError(f"không thấy pair_id {pair_id!r} trong {path}")


def sample_to_dict(sample: Sample) -> dict:
    return asdict(sample)
