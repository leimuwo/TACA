"""Shared bi-encoder configuration and dependency-safe training primitives."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
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


def _mean_pool(outputs: Any, attention_mask: Any, torch_module: ModuleType) -> Any:
    hidden = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return torch_module.nn.functional.normalize(pooled, p=2, dim=1)


def _encode_texts(
    texts: list[str],
    *,
    tokenizer: Any,
    model: Any,
    max_length: int,
    torch_module: ModuleType,
) -> Any:
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    outputs = model(**encoded)
    return _mean_pool(outputs, encoded["attention_mask"], torch_module)


def train_one_batch(
    examples: tuple[TrainingExample, ...],
    *,
    tokenizer: Any,
    model: Any,
    optimizer: Any,
    config: TrainingConfig,
    torch_module: ModuleType,
) -> float:
    """Run the pilot InfoNCE plus explicit hard-negative margin objective once."""

    if not examples:
        raise ValueError("one-batch training requires examples")
    batch = examples[: config.batch_size]
    model.train()
    optimizer.zero_grad()
    intent_embeddings = _encode_texts(
        [example.intent for example in batch],
        tokenizer=tokenizer,
        model=model,
        max_length=config.intent_max_length,
        torch_module=torch_module,
    )
    positive_embeddings = _encode_texts(
        [example.positive for example in batch],
        tokenizer=tokenizer,
        model=model,
        max_length=config.action_max_length,
        torch_module=torch_module,
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
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def evaluate_encoder(
    examples: tuple[TrainingExample, ...],
    *,
    tokenizer: Any,
    model: Any,
    config: TrainingConfig,
    torch_module: ModuleType,
    method: str,
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
    torch_module.manual_seed(config.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model)
    model = transformers.AutoModel.from_pretrained(config.base_model)
    model.to("cpu")
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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
        )
        for split, examples in split_examples.items()
    }
    best_mrr = -1.0
    best_epoch = 0
    epoch_reports: list[dict[str, Any]] = []
    train_examples = split_examples["train"]
    for epoch in range(1, config.epochs + 1):
        losses = []
        for start in range(0, len(train_examples), config.batch_size):
            losses.append(
                train_one_batch(
                    train_examples[start : start + config.batch_size],
                    tokenizer=tokenizer,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    torch_module=torch_module,
                )
            )
        validation = evaluate_encoder(
            split_examples["validation"],
            tokenizer=tokenizer,
            model=model,
            config=config,
            torch_module=torch_module,
            method="fine-tuned-bge",
        )
        epoch_report = {
            "epoch": epoch,
            "mean_loss": sum(losses) / len(losses),
            "validation": validation,
        }
        epoch_reports.append(epoch_report)
        if validation["retrieval"]["mrr"] > best_mrr:
            best_mrr = validation["retrieval"]["mrr"]
            best_epoch = epoch
            checkpoint_dir = output_dir / "checkpoint-best"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            (checkpoint_dir / "metadata.json").write_text(
                json.dumps(
                    build_checkpoint_metadata(
                        base_model=config.base_model,
                        epoch=epoch,
                        validation_mrr=best_mrr,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    selected = {
        split: evaluate_encoder(
            examples,
            tokenizer=tokenizer,
            model=model,
            config=config,
            torch_module=torch_module,
            method="fine-tuned-bge",
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
        },
        "frozen": frozen,
        "fine_tuned": selected,
        "epoch_reports": epoch_reports,
        "best_epoch": best_epoch,
        "supervision_status": "provisional_auto_pair",
        "experiment_tier": "feasibility_only",
        "human_reviewed": False,
    }
