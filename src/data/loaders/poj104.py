"""POJ-104 loader (C/C++) — dataset duy nhất cho C/C++ ở core experiment.

Nguồn: https://huggingface.co/datasets/google/code_x_glue_cc_clone_detection_poj104
Schema dự kiến (CodeXGLUE): train/valid/test với các cột như
`func`, `label`/`problem_id`, `idx`. Schema thực tế được xác nhận ở
bước audit (reports/002) và cập nhật vào describe_schema().
"""

from datasets import load_dataset

from src.data.loaders.base import CodeSample, DatasetAdapter


class POJ104Loader(DatasetAdapter):
    language = "cpp"
    HF_ID = "google/code_x_glue_cc_clone_detection_poj104"

    def load(self, **kwargs) -> list[CodeSample]:
        ds = load_dataset(self.HF_ID, **kwargs)
        samples: list[CodeSample] = []
        for split in ds:
            for i, row in enumerate(ds[split]):
                samples.append(
                    CodeSample(
                        sample_id=f"poj104-{split}-{i}",
                        code=row.get("func", row.get("code", "")),
                        problem_id=str(row.get("label", row.get("problem_id", "?"))),
                        language=self.language,
                        split=str(split),
                        metadata={"row_keys": list(row.keys())},
                    )
                )
        return samples

    def describe_schema(self) -> dict:
        return {
            "hf_id": self.HF_ID,
            "language": self.language,
            "expected_splits": ["train", "validation", "test"],
            "expected_fields": ["func/code", "label/problem_id"],
            "note": "Schema thực tế chốt sau audit reports/002_dataset_audit_poj104.md",
        }
