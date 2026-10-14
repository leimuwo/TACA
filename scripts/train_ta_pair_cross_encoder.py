#!/usr/bin/env python3
"""Fine-tune the pretrained BGE reranker as a TA pair Cross-Encoder."""
import os
from pathlib import Path

from train_ta_pair_reranker import main


if __name__ == "__main__":
    main(default_model=Path(os.environ.get("TA_MODEL_ROOT", "../model_hub")) / Path("bge-reranker-v2-m3"),
         architecture="pretrained_bge_reranker_cross_encoder",
         classification_head="pretrained_one_logit")
