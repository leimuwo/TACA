#!/usr/bin/env python3
"""Validate or train the provisional shared Intent--Action bi-encoder."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from thought_action_retrieval.training.baselines import FEASIBILITY_LABEL
from thought_action_retrieval.training.biencoder import (
    TrainingConfig,
    load_training_stack,
    prepare_training_examples,
    train_shared_biencoder,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing training split: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"training split must contain JSON objects: {path}")
    return rows


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _publish_new_directory(output_dir: Path, files: dict[str, bytes]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for relative, content in files.items():
            path = staged / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        staged.replace(output_dir)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def run(args: argparse.Namespace) -> int:
    config = TrainingConfig(
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
    )
    split_examples = {
        split: prepare_training_examples(
            _read_jsonl(args.dataset_dir.resolve() / "splits" / f"{split}.jsonl")
        )
        for split in ("train", "validation", "test")
    }
    plan = {
        "schema_version": "provisional_biencoder_training_plan_v1",
        "label": FEASIBILITY_LABEL,
        "split_counts": {
            split: len(examples) for split, examples in sorted(split_examples.items())
        },
        "config": asdict(config),
        "supervision_status": "provisional_auto_pair",
        "experiment_tier": "feasibility_only",
        "human_reviewed": False,
    }
    if args.dry_run:
        _publish_new_directory(
            args.output_dir.resolve(), {"training_plan.json": _json_bytes(plan)}
        )
        print(f"validated provisional training plan in {args.output_dir.resolve()}")
        return 0
    result = train_shared_biencoder(
        split_examples=split_examples,
        config=config,
        output_dir=args.output_dir,
    )
    (args.output_dir.resolve() / "training_report.json").write_bytes(
        _json_bytes(result)
    )
    print(f"wrote provisional bi-encoder report to {args.output_dir.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="single-device runtime; auto selects CUDA when available",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp32"),
        default="auto",
        help="auto selects BF16 on CUDA devices that support it",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
