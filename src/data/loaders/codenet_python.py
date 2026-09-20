"""CodeNet Python loader — dataset duy nhất cho Python ở core experiment.

Nguồn: https://huggingface.co/datasets/windchimeran/codenet_python
Nếu dataset thiếu split cho clone detection: split theo problem/task
identity (không random theo source) để tránh leakage.
"""

from datasets import load_dataset

from src.data.loaders.base import CodeSample, DatasetAdapter


class CodeNetPythonLoader(DatasetAdapter):
    language = "python"
    HF_ID = "windchimeran/codenet_python"

    def load(
        self,
        split: str | None = None,
        streaming: bool = False,
        max_samples: int = 0,
        shuffle_buffer: int = 0,
        seed: int = 42,
        **kwargs,
    ) -> list[CodeSample]:
        """Nạp CodeNet Python.

        - `split`: chỉ nạp split này (`small`/`big`); mặc định nạp tất cả.
        - `streaming`: bật khi dataset quá lớn cho arrow cache Windows
          (DEC-008); kết hợp `shuffle_buffer` + `max_samples` để lấy mẫu
          không bias mà không cần generate toàn bộ 4.4M mẫu.
        """
        ds = load_dataset(self.HF_ID, streaming=streaming, **kwargs)
        names = [split] if split else list(ds.keys())
        samples: list[CodeSample] = []
        counter = 0
        for name in names:
            iterable = ds[name]
            if streaming and shuffle_buffer > 0:
                iterable = iterable.shuffle(seed=seed, buffer_size=shuffle_buffer)
            for i, row in enumerate(iterable):
                if max_samples and len(samples) >= max_samples:
                    break
                code = row.get("code", row.get("source_code", row.get("func", "")))
                problem = row.get(
                    "problem_id", row.get("task_id", row.get("label", "?"))
                )
                samples.append(
                    CodeSample(
                        sample_id=f"codenet-py-{name}-{counter}",
                        code=code,
                        problem_id=str(problem),
                        language=self.language,
                        split=str(name),
                        metadata={
                            "row_keys": list(row.keys()),
                            "status": str(row.get("status", "?")),
                            "submission_id": str(row.get("submission_id", "")),
                            "code_size": row.get("code_size"),
                        },
                    )
                )
                counter += 1
            if max_samples and len(samples) >= max_samples:
                break
        return samples

    def describe_schema(self) -> dict:
        return {
            "hf_id": self.HF_ID,
            "language": self.language,
            "split_policy": "theo problem/task identity nếu thiếu split chuẩn",
            "note": "Schema thực tế chốt sau audit reports/003_dataset_audit_codenet_python.md",
        }
