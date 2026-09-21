"""Shared bi-encoder configuration and dependency-safe training primitives."""

from __future__ import annotations

import importlib
import json
import math
import random
import shutil
from contextlib import nullcontext
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from thought_action_retrieval.training.baselines import FEASIBILITY_LABEL
from thought_action_retrieval.training.metrics import pairwise_metrics, retrieval_metrics


@dataclass(frozen=True)
class TrainingConfig:
    base_model: str = "BAAI/bge-small-en-v1.5"
    intent_max_length: int = 128
    action_max_length: int = 192
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    temperature: float = 0.05
    margin: float = 0.2
    hard_negative_weight: float = 0.5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 20260921
    device: str = "auto"
    precision: str = "auto"
    log_every_steps: int = 10

    def __post_init__(self) -> None:
        if not self.base_model.strip():
            raise ValueError("base_model cannot be empty")
        if self.epochs < 1 or self.epochs > 3:
            raise ValueError("epochs must be between 1 and 3 for the pilot")
        for name in (
            "intent_max_length",
            "action_max_length",
            "batch_size",
            "gradient_accumulation_steps",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("learning_rate", "temperature"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.margin < 0 or self.hard_negative_weight < 0:
            raise ValueError("margin and hard_negative_weight cannot be negative")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError("device must be auto, cuda, or cpu")
        if self.precision not in {"auto", "bf16", "fp32"}:
            raise ValueError("precision must be auto, bf16, or fp32")
        if self.log_every_steps < 1:
            raise ValueError("log_every_steps must be positive")


@dataclass(frozen=True)
class TrainingRuntime:
    device: str
    precision: str
    tf32_enabled: bool


def resolve_runtime(
    torch_module: ModuleType,
    *,
    device_request: str,
    precision_request: str,
) -> TrainingRuntime:
    """Resolve a fail-fast single-device runtime suitable for an H100 host."""

    if device_request not in {"auto", "cuda", "cpu"}:
        raise ValueError("device_request must be auto, cuda, or cpu")
    if precision_request not in {"auto", "bf16", "fp32"}:
        raise ValueError("precision_request must be auto, bf16, or fp32")
    cuda_available = bool(torch_module.cuda.is_available())
    device = "cuda" if device_request == "auto" and cuda_available else device_request
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested but no CUDA device is visible; check the H100 "
            "allocation and CUDA-enabled PyTorch installation"
        )
    bf16_supported = bool(
        device == "cuda"
        and getattr(torch_module.cuda, "is_bf16_supported", lambda: False)()
    )
    if precision_request == "auto":
        precision = "bf16" if bf16_supported else "fp32"
    else:
        precision = precision_request
    if precision == "bf16" and not bf16_supported:
        raise RuntimeError(
            "BF16 was requested but the selected CUDA device or PyTorch build "
            "does not report BF16 support"
        )
    return TrainingRuntime(
        device=device,
        precision=precision,
        tf32_enabled=device == "cuda",
    )


def configure_runtime(
    torch_module: ModuleType, runtime: TrainingRuntime, *, seed: int
) -> None:
    """Configure deterministic seeds and H100-friendly CUDA math settings."""

    torch_module.manual_seed(seed)
    if runtime.device != "cuda":
        return
    torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.allow_tf32 = True
    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class TrainingExample:
    candidate_id: str
    intent: str
    positive: str
    negatives: tuple[str, ...]


def prepare_training_examples(
    records: Iterable[dict[str, Any]],
) -> tuple[TrainingExample, ...]:
    """Validate permanent pilot watermarks and materialize encoder examples."""

    examples: list[TrainingExample] = []
    for row in records:
        if (
            row.get("supervision_status") != "provisional_auto_pair"
            or row.get("experiment_tier") != "feasibility_only"
            or row.get("human_reviewed") is not False
        ):
            raise ValueError("training record is missing provisional watermarks")
        candidate_id = str(row.get("candidate_id", "")).strip()
        intent = str(row.get("intent_input", "")).strip()
        positive_action = row.get("positive_action")
        positive = (
            str(positive_action.get("serialized", "")).strip()
            if isinstance(positive_action, dict)
            else ""
        )
        negatives = tuple(
            str(item.get("serialized", "")).strip()
            for item in row.get("negative_actions", [])
            if isinstance(item, dict) and str(item.get("serialized", "")).strip()
        )
        if not candidate_id or not intent.startswith("[INTENT] "):
            raise ValueError("training example requires candidate_id and [INTENT] input")
        if not positive.startswith("[ACTION] "):
            raise ValueError("training example requires a serialized [ACTION] positive")
        if any(not negative.startswith("[ACTION] ") for negative in negatives):
            raise ValueError("hard negatives must use serialized [ACTION] inputs")
        if positive in negatives or len(set(negatives)) != len(negatives):
            raise ValueError("hard negatives must be unique and differ from the positive")
        examples.append(
            TrainingExample(
                candidate_id=candidate_id,
                intent=intent,
                positive=positive,
                negatives=negatives,
            )
        )
    if not examples:
        raise ValueError("training split cannot be empty")
    return tuple(examples)


def load_training_stack(
    importer: Callable[[str], ModuleType] = importlib.import_module,
) -> dict[str, ModuleType]:
    """Import heavy dependencies lazily so ordinary repository tests stay light."""

    modules: dict[str, ModuleType] = {}
    try:
        for name in ("numpy", "torch", "transformers"):
            modules[name] = importer(name)
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "bi-encoder dependencies are unavailable; install the project with "
            "the [training] extra in the repository-local .venv"
        ) from error
    return modules


def build_checkpoint_metadata(
    *,
    base_model: str,
    epoch: int,
    validation_mrr: float,
) -> dict[str, Any]:
    if epoch < 0 or not 0 <= validation_mrr <= 1:
        raise ValueError("invalid checkpoint epoch or validation MRR")
    return {
        "schema_version": "provisional_biencoder_checkpoint_v1",
        "label": FEASIBILITY_LABEL,
        "base_model": base_model,
        "epoch": epoch,
        "validation_mrr": validation_mrr,
        "supervision_status": "provisional_auto_pair",
        "experiment_tier": "feasibility_only",
        "human_reviewed": False,
    }


def _save_checkpoint(
    directory: Path,
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    torch_module: ModuleType,
    config: TrainingConfig,
    runtime: TrainingRuntime,
    epoch: int,
    validation_mrr: float,
    best_epoch: int,
) -> None:
    staging = directory.with_name(f".{directory.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging)
    tokenizer.save_pretrained(staging)
    torch_module.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_mrr": validation_mrr,
            "config": config.__dict__,
            "runtime": runtime.__dict__,
        },
        staging / "trainer_state.pt",
    )
    (staging / "metadata.json").write_text(
        json.dumps(
            {
                **build_checkpoint_metadata(
                    base_model=config.base_model,
                    epoch=epoch,
                    validation_mrr=validation_mrr,
                ),
                "runtime": runtime.__dict__,
                "best_epoch": best_epoch,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if directory.exists():
        shutil.rmtree(directory)
    staging.replace(directory)


def _mean_pool(outputs: Any, attention_mask: Any, torch_module: ModuleType) -> Any:
    hidden = outputs.last_hidden_state.float()
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch_module.nn.functional.normalize(pooled, p=2, dim=1)


def _autocast_context(
    torch_module: ModuleType, *, device: str, precision: str
) -> Any:
    if precision == "bf16":
        return torch_module.autocast(
            device_type="cuda", dtype=torch_module.bfloat16
        )
    return nullcontext()


def _encode_texts(
    texts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    max_length: int,
    torch_module: ModuleType,
    precision: str = "fp32",
) -> Any:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in encoded.items()
    }
    with _autocast_context(
        torch_module, device=device.type, precision=precision
    ):
        outputs = model(**encoded)
    return _mean_pool(outputs, encoded["attention_mask"], torch_module)


def _compute_batch_loss(
    batch: tuple[TrainingExample, ...],
    *,
    tokenizer: Any,
    model: Any,
    config: TrainingConfig,
    torch_module: ModuleType,
    precision: str,
) -> Any:
    intent_embeddings = _encode_texts(
        [example.intent for example in batch],
        tokenizer=tokenizer,
        model=model,
        max_length=config.intent_max_length,
        torch_module=torch_module,
        precision=precision,
    )
    positive_embeddings = _encode_texts(
        [example.positive for example in batch],
        tokenizer=tokenizer,
        model=model,
        max_length=config.action_max_length,
        torch_module=torch_module,
        precision=precision,
    )
    logits = intent_embeddings @ positive_embeddings.transpose(0, 1)
    logits = logits / config.temperature
    labels = torch_module.arange(len(batch), device=logits.device)
    nce_loss = torch_module.nn.functional.cross_entropy(logits, labels)

    margin_terms = []
    for index, example in enumerate(batch):
        if not example.negatives:
            continue
        negative_embeddings = _encode_texts(
            list(example.negatives),
            tokenizer=tokenizer,
            model=model,
            max_length=config.action_max_length,
            torch_module=torch_module,
            precision=precision,
        )
        positive_score = (intent_embeddings[index] * positive_embeddings[index]).sum()
        negative_scores = negative_embeddings @ intent_embeddings[index]
        margin_terms.append(
            torch_module.relu(config.margin - positive_score + negative_scores).mean()
        )
    hard_loss = (
        torch_module.stack(margin_terms).mean()
        if margin_terms
        else torch_module.zeros((), device=logits.device)
    )
    loss = nce_loss + config.hard_negative_weight * hard_loss
    if not bool(torch_module.isfinite(loss)):
        raise FloatingPointError("training loss is not finite")
    return loss


def train_one_batch(
    examples: tuple[TrainingExample, ...],
    *,
    tokenizer: Any,
    model: Any,
    optimizer: Any,
    config: TrainingConfig,
    torch_module: ModuleType,
    precision: str = "fp32",
) -> float:
    """Run the pilot InfoNCE plus explicit hard-negative margin objective once."""

    if not examples:
        raise ValueError("one-batch training requires examples")
    batch = examples[: config.batch_size]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = _compute_batch_loss(
        batch,
        tokenizer=tokenizer,
        model=model,
        config=config,
        torch_module=torch_module,
        precision=precision,
    )
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def train_one_epoch(
    examples: tuple[TrainingExample, ...],
    *,
    tokenizer: Any,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: TrainingConfig,
    torch_module: ModuleType,
    precision: str,
    epoch: int,
) -> dict[str, Any]:
    """Train one deterministically shuffled epoch with true gradient accumulation."""

    if not examples:
        raise ValueError("one-epoch training requires examples")
    ordered = list(examples)
    random.Random(config.seed + epoch).shuffle(ordered)
    micro_batches = [
        tuple(ordered[start : start + config.batch_size])
        for start in range(0, len(ordered), config.batch_size)
    ]
    losses: list[float] = []
    optimizer_steps = 0
    model.train()
    for group_start in range(
        0, len(micro_batches), config.gradient_accumulation_steps
    ):
        accumulation_group = micro_batches[
            group_start : group_start + config.gradient_accumulation_steps
        ]
        optimizer.zero_grad(set_to_none=True)
        for batch in accumulation_group:
            loss = _compute_batch_loss(
                batch,
                tokenizer=tokenizer,
                model=model,
                config=config,
                torch_module=torch_module,
                precision=precision,
            )
            losses.append(float(loss.detach().cpu()))
            (loss / len(accumulation_group)).backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer_steps += 1
        if (
            optimizer_steps % config.log_every_steps == 0
            or group_start + len(accumulation_group) == len(micro_batches)
        ):
            print(
                json.dumps(
                    {
                        "event": "training_progress",
                        "epoch": epoch,
                        "optimizer_step": optimizer_steps,
                        "optimizer_steps_in_epoch": math.ceil(
                            len(micro_batches)
                            / config.gradient_accumulation_steps
                        ),
                        "latest_loss": losses[-1],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return {
        "mean_loss": sum(losses) / len(losses),
        "micro_batches": len(micro_batches),
        "optimizer_steps": optimizer_steps,
    }


def evaluate_encoder(
    examples: tuple[TrainingExample, ...],
    *,
    tokenizer: Any,
    model: Any,
    config: TrainingConfig,
    torch_module: ModuleType,
    method: str,
    precision: str = "fp32",
) -> dict[str, Any]:
    """Score each positive and explicit negative using one shared encoder."""

    if not examples:
        raise ValueError("encoder evaluation requires examples")
    model.eval()
    with torch_module.no_grad():
        intents = _encode_texts(
            [example.intent for example in examples],
            tokenizer=tokenizer,
            model=model,
            max_length=config.intent_max_length,
            torch_module=torch_module,
            precision=precision,
        )
        rankings: list[list[str]] = []
        positive_ids: list[set[str]] = []
        positive_scores: list[float] = []
        negative_scores: list[list[float]] = []
        for index, example in enumerate(examples):
            actions = (example.positive, *example.negatives)
            action_embeddings = _encode_texts(
                list(actions),
                tokenizer=tokenizer,
                model=model,
                max_length=config.action_max_length,
                torch_module=torch_module,
                precision=precision,
            )
            scores = action_embeddings @ intents[index]
            ordered = sorted(
                zip(scores.detach().cpu().tolist(), actions, strict=True),
                key=lambda item: (-item[0], item[1]),
            )
            rankings.append([action for _, action in ordered])
            positive_ids.append({example.positive})
            positive_scores.append(float(scores[0].detach().cpu()))
            negative_scores.append(
                [float(score.detach().cpu()) for score in scores[1:]]
            )
    return {
        "label": FEASIBILITY_LABEL,
        "method": method,
        "retrieval": retrieval_metrics(rankings, positive_ids),
        "hard_negative": pairwise_metrics(positive_scores, negative_scores),
        "document_count": sum(1 + len(example.negatives) for example in examples),
    }


def train_shared_biencoder(
    split_examples: dict[str, tuple[TrainingExample, ...]],
    *,
    config: TrainingConfig,
    output_dir: Any,
) -> dict[str, Any]:
    """Download/load one shared encoder, train it, and emit feasibility reports."""

    modules = load_training_stack()
    torch_module = modules["torch"]
    transformers = modules["transformers"]
    runtime = resolve_runtime(
        torch_module,
        device_request=config.device,
        precision_request=config.precision,
    )
    configure_runtime(torch_module, runtime, seed=config.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model)
    model = transformers.AutoModel.from_pretrained(config.base_model)
    model.to(runtime.device)
    optimizer_kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if runtime.device == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch_module.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch_module.optim.AdamW(model.parameters(), **optimizer_kwargs)
    optimizer_steps_per_epoch = math.ceil(
        math.ceil(len(split_examples["train"]) / config.batch_size)
        / config.gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * config.epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    if hasattr(transformers, "get_linear_schedule_with_warmup"):
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
    else:
        scheduler = torch_module.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        split: evaluate_encoder(
            examples,
            tokenizer=tokenizer,
            model=model,
            config=config,
            torch_module=torch_module,
            method="frozen-bge",
            precision=runtime.precision,
        )
        for split, examples in split_examples.items()
    }
    best_mrr = -1.0
    best_epoch = 0
    epoch_reports: list[dict[str, Any]] = []
    train_examples = split_examples["train"]
    for epoch in range(1, config.epochs + 1):
        train_report = train_one_epoch(
            train_examples,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            torch_module=torch_module,
            precision=runtime.precision,
            epoch=epoch,
        )
        validation = evaluate_encoder(
            split_examples["validation"],
            tokenizer=tokenizer,
            model=model,
            config=config,
            torch_module=torch_module,
            method="fine-tuned-bge",
            precision=runtime.precision,
        )
        epoch_report = {
            "epoch": epoch,
            **train_report,
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        epoch_reports.append(epoch_report)
        if validation["retrieval"]["mrr"] > best_mrr:
            best_mrr = validation["retrieval"]["mrr"]
            best_epoch = epoch
            _save_checkpoint(
                output_dir / "checkpoint-best",
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                torch_module=torch_module,
                config=config,
                runtime=runtime,
                epoch=epoch,
                validation_mrr=best_mrr,
                best_epoch=best_epoch,
            )
        _save_checkpoint(
            output_dir / "checkpoint-last",
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            torch_module=torch_module,
            config=config,
            runtime=runtime,
            epoch=epoch,
            validation_mrr=best_mrr,
            best_epoch=best_epoch,
        )
    best_dir = output_dir / "checkpoint-best"
    if not best_dir.is_dir():
        raise RuntimeError("training did not produce checkpoint-best")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(best_dir))
    model = transformers.AutoModel.from_pretrained(str(best_dir))
    model.to(runtime.device)
    selected = {
        split: evaluate_encoder(
            examples,
            tokenizer=tokenizer,
            model=model,
            config=config,
            torch_module=torch_module,
            method="fine-tuned-bge",
            precision=runtime.precision,
        )
        for split, examples in split_examples.items()
    }
    return {
        "schema_version": "provisional_biencoder_pilot_report_v1",
        "label": FEASIBILITY_LABEL,
        "config": {
            "base_model": config.base_model,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "warmup_ratio": config.warmup_ratio,
            "seed": config.seed,
        },
        "runtime": runtime.__dict__,
        "frozen": frozen,
        "fine_tuned": selected,
        "epoch_reports": epoch_reports,
        "best_epoch": best_epoch,
        "selected_checkpoint": "checkpoint-best",
        "supervision_status": "provisional_auto_pair",
        "experiment_tier": "feasibility_only",
        "human_reviewed": False,
    }
