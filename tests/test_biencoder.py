import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from thought_action_retrieval.training.biencoder import (
    TrainingConfig,
    build_checkpoint_metadata,
    evaluate_encoder,
    load_training_stack,
    prepare_training_examples,
    train_one_batch,
)


def _record(candidate_id="one", split="train"):
    return {
        "candidate_id": candidate_id,
        "split": split,
        "intent_input": "[INTENT] Go to line 550.",
        "positive_action": {"serialized": "[ACTION] goto(line=550)"},
        "negative_actions": [
            {
                "negative_id": f"{candidate_id}:N1",
                "serialized": "[ACTION] goto(line=450)",
            }
        ],
        "supervision_status": "provisional_auto_pair",
        "experiment_tier": "feasibility_only",
        "human_reviewed": False,
    }


class BiEncoderConfigurationTests(unittest.TestCase):
    @staticmethod
    def _tiny_stack(torch):
        class TinyTokenizer:
            def __call__(self, texts, **kwargs):
                width = kwargs["max_length"]
                rows = []
                masks = []
                for text in texts:
                    values = [1 + (ord(char) % 7) for char in text[: width - 2]]
                    values = [1] + values + [2]
                    values = values[:width]
                    mask = [1] * len(values)
                    values += [0] * (width - len(values))
                    mask += [0] * (width - len(mask))
                    rows.append(values)
                    masks.append(mask)
                return {
                    "input_ids": torch.tensor(rows),
                    "attention_mask": torch.tensor(masks),
                }

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)

            def forward(self, input_ids, attention_mask):
                return type(
                    "Output",
                    (),
                    {"last_hidden_state": self.embedding(input_ids)},
                )()

        return TinyTokenizer(), TinyEncoder()

    def test_rejects_unsafe_or_invalid_training_configuration(self):
        for kwargs in (
            {"epochs": 0},
            {"epochs": 4},
            {"batch_size": 0},
            {"learning_rate": 0},
            {"temperature": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TrainingConfig(**kwargs)

    def test_prepares_prefixed_positive_and_hard_negative_examples(self):
        examples = prepare_training_examples([_record()])

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].intent, "[INTENT] Go to line 550.")
        self.assertEqual(examples[0].positive, "[ACTION] goto(line=550)")
        self.assertEqual(examples[0].negatives, ("[ACTION] goto(line=450)",))

    def test_rejects_records_without_permanent_provisional_watermarks(self):
        record = _record()
        record["human_reviewed"] = True

        with self.assertRaisesRegex(ValueError, "provisional"):
            prepare_training_examples([record])

    def test_dependency_diagnostic_names_install_extra_without_importing_eagerly(self):
        def unavailable(name):
            raise ModuleNotFoundError(name)

        with self.assertRaisesRegex(RuntimeError, r"\[training\]"):
            load_training_stack(importer=unavailable)

    def test_checkpoint_metadata_is_visibly_feasibility_only(self):
        metadata = build_checkpoint_metadata(
            base_model="BAAI/bge-small-en-v1.5",
            epoch=2,
            validation_mrr=0.75,
        )

        self.assertEqual(
            metadata["label"], "FEASIBILITY ONLY — NOT GOLD EVALUATION"
        )
        self.assertEqual(metadata["supervision_status"], "provisional_auto_pair")
        self.assertEqual(metadata["epoch"], 2)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "training dependency smoke test"
    )
    def test_one_batch_smoke_updates_a_shared_encoder_without_downloads(self):
        import torch
        tokenizer, model = self._tiny_stack(torch)
        before = model.embedding.weight.detach().clone()
        loss = train_one_batch(
            prepare_training_examples([_record()]),
            tokenizer=tokenizer,
            model=model,
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            config=TrainingConfig(batch_size=1, gradient_accumulation_steps=1),
            torch_module=torch,
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.embedding.weight.detach()))

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "training dependency smoke test"
    )
    def test_encoder_evaluation_emits_shared_retrieval_and_pairwise_schema(self):
        import torch

        tokenizer, model = self._tiny_stack(torch)
        report = evaluate_encoder(
            prepare_training_examples([_record()]),
            tokenizer=tokenizer,
            model=model,
            config=TrainingConfig(batch_size=1),
            torch_module=torch,
            method="tiny-frozen",
        )

        self.assertEqual(report["method"], "tiny-frozen")
        self.assertEqual(report["retrieval"]["query_count"], 1)
        self.assertEqual(report["hard_negative"]["comparison_count"], 1)
        self.assertEqual(
            report["label"], "FEASIBILITY ONLY — NOT GOLD EVALUATION"
        )


class BiEncoderCliTests(unittest.TestCase):
    def test_dry_run_validates_all_splits_without_loading_model_dependencies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "experiment"
            (dataset / "splits").mkdir(parents=True)
            for split in ("train", "validation", "test"):
                row = _record(candidate_id=split, split=split)
                (dataset / "splits" / f"{split}.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_biencoder.py",
                    "--dataset-dir",
                    str(dataset),
                    "--output-dir",
                    str(output),
                    "--dry-run",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(
                (output / "training_plan.json").read_text(encoding="utf-8")
            )
            self.assertEqual(plan["split_counts"], {"test": 1, "train": 1, "validation": 1})
            self.assertEqual(
                plan["label"], "FEASIBILITY ONLY — NOT GOLD EVALUATION"
            )


if __name__ == "__main__":
    unittest.main()
