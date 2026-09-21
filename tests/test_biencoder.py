import json
import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import thought_action_retrieval.training.biencoder as biencoder_module
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

    @staticmethod
    def _tiny_transformers(torch):
        load_sources = []

        class TinyTokenizer:
            def __call__(self, texts, **kwargs):
                width = kwargs["max_length"]
                rows = []
                masks = []
                for text in texts:
                    values = [1 + (ord(char) % 7) for char in text[: width - 2]]
                    values = ([1] + values + [2])[:width]
                    mask = [1] * len(values)
                    values += [0] * (width - len(values))
                    mask += [0] * (width - len(mask))
                    rows.append(values)
                    masks.append(mask)
                return {
                    "input_ids": torch.tensor(rows),
                    "attention_mask": torch.tensor(masks),
                }

            def save_pretrained(self, path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                (path / "tiny_tokenizer.json").write_text("{}\n", encoding="utf-8")

        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(8, 4)

            def forward(self, input_ids, attention_mask):
                return types.SimpleNamespace(
                    last_hidden_state=self.embedding(input_ids)
                )

            def save_pretrained(self, path):
                path = Path(path)
                path.mkdir(parents=True, exist_ok=True)
                torch.save(self.state_dict(), path / "tiny_model.pt")

        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, source):
                return TinyTokenizer()

        class AutoModel:
            @classmethod
            def from_pretrained(cls, source):
                load_sources.append(str(source))
                model = TinyEncoder()
                weights = Path(str(source)) / "tiny_model.pt"
                if weights.is_file():
                    model.load_state_dict(torch.load(weights, map_location="cpu"))
                return model

        return types.SimpleNamespace(
            AutoTokenizer=AutoTokenizer,
            AutoModel=AutoModel,
            get_linear_schedule_with_warmup=lambda optimizer, **kwargs: (
                torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            ),
            load_sources=load_sources,
        )

    def test_rejects_unsafe_or_invalid_training_configuration(self):
        for kwargs in (
            {"epochs": 0},
            {"epochs": 4},
            {"batch_size": 0},
            {"learning_rate": 0},
            {"temperature": 0},
            {"device": "tpu"},
            {"precision": "fp16"},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TrainingConfig(**kwargs)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "training dependency smoke test"
    )
    def test_auto_runtime_selects_cuda_bf16_for_an_h100_capable_host(self):
        import torch

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True),
        ):
            runtime = biencoder_module.resolve_runtime(
                torch, device_request="auto", precision_request="auto"
            )

        self.assertEqual(runtime.device, "cuda")
        self.assertEqual(runtime.precision, "bf16")
        self.assertTrue(runtime.tf32_enabled)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "training dependency smoke test"
    )
    def test_explicit_cuda_fails_fast_when_no_gpu_is_visible(self):
        import torch

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                biencoder_module.resolve_runtime(
                    torch, device_request="cuda", precision_request="auto"
                )

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
    def test_train_epoch_steps_only_at_gradient_accumulation_boundaries(self):
        import torch

        class CountingSGD(torch.optim.SGD):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.step_count = 0

            def step(self, closure=None):
                self.step_count += 1
                return super().step(closure)

        tokenizer, model = self._tiny_stack(torch)
        examples = prepare_training_examples(
            [_record(candidate_id=f"case-{index}") for index in range(3)]
        )
        optimizer = CountingSGD(model.parameters(), lr=0.1)

        report = biencoder_module.train_one_epoch(
            examples,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            config=TrainingConfig(
                batch_size=1,
                gradient_accumulation_steps=2,
                seed=7,
            ),
            torch_module=torch,
            precision="fp32",
            epoch=1,
        )

        self.assertEqual(optimizer.step_count, 2)
        self.assertEqual(report["optimizer_steps"], 2)
        self.assertEqual(report["micro_batches"], 3)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "training dependency smoke test"
    )
    def test_full_training_reloads_best_checkpoint_for_final_evaluation(self):
        import torch

        transformers = self._tiny_transformers(torch)
        examples = prepare_training_examples([_record()])
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(
                biencoder_module,
                "load_training_stack",
                return_value={"torch": torch, "transformers": transformers},
            ):
                report = biencoder_module.train_shared_biencoder(
                    split_examples={
                        "train": examples,
                        "validation": examples,
                        "test": examples,
                    },
                    config=TrainingConfig(
                        epochs=1,
                        batch_size=1,
                        gradient_accumulation_steps=1,
                        device="cpu",
                        precision="fp32",
                    ),
                    output_dir=Path(temporary),
                )

            best_path = str(Path(temporary) / "checkpoint-best")
            self.assertEqual(transformers.load_sources[-1], best_path)
            self.assertEqual(report["selected_checkpoint"], "checkpoint-best")
            self.assertEqual(report["runtime"]["device"], "cpu")
            self.assertTrue(
                (Path(temporary) / "checkpoint-last" / "trainer_state.pt").is_file()
            )

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

    def test_dry_run_records_h100_runtime_and_accumulation_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "experiment"
            (dataset / "splits").mkdir(parents=True)
            for split in ("train", "validation", "test"):
                (dataset / "splits" / f"{split}.jsonl").write_text(
                    json.dumps(_record(candidate_id=split, split=split)) + "\n",
                    encoding="utf-8",
                )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_biencoder.py",
                    "--dataset-dir",
                    str(dataset),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cuda",
                    "--precision",
                    "bf16",
                    "--gradient-accumulation-steps",
                    "2",
                    "--seed",
                    "17",
                    "--dry-run",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads(
                (output / "training_plan.json").read_text(encoding="utf-8")
            )["config"]
            self.assertEqual(config["device"], "cuda")
            self.assertEqual(config["precision"], "bf16")
            self.assertEqual(config["gradient_accumulation_steps"], 2)
            self.assertEqual(config["seed"], 17)


if __name__ == "__main__":
    unittest.main()
