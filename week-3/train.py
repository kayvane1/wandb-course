import argparse
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_metric, load_from_disk
import numpy as np
import pandas as pd
from ml_collections import config_dict
import wandb
import os
from wandb.beta.workflows import log_model, use_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# defaults
default_cfg = config_dict.ConfigDict()

# WANDB BASE PARAMETERS
default_cfg.PROJECT_NAME = "wandb-week-3-complaints-classifier"
# WANDB ARTIFACT TYPES
default_cfg.DATASET_TYPE = "dataset"
default_cfg.MODEL_TYPE = "model"
# WANDB JOB TYPES
default_cfg.MODEL_TRAINING_JOB_TYPE = "model-training"
# WANDB ARTIFACT NAMES
default_cfg.TRAIN_DATA_ARTIFACT = "complaints_train_data"
default_cfg.TEST_DATA_ARTIFACT = "complaints_test_data"
# DATA FOLDERS
default_cfg.TRAIN_DATA_FOLDER = "complaints-dataset/train"
default_cfg.TEST_DATA_FOLDER = "complaints-dataset/test"
default_cfg.MODEL_DATA_FOLDER = "complaints-model"
# TRANSFORMERS PARAMETERS
default_cfg.model_name = "distilbert-base-uncased"
default_cfg.num_epochs = 2
default_cfg.train_batch_size = 32
default_cfg.eval_batch_size = 32
default_cfg.warmup_steps = 1500
default_cfg.learning_rate = 5e-5
default_cfg.fp16 = True
# HUB PARAMETERS
default_cfg.PUSH_TO_HUB = True
default_cfg.HUB_STRATEGY = "every_save"


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(
        description="Process base parameters & hyper-parameters"
    )
    argparser.add_argument(
        "--model_name",
        type=str,
        default=default_cfg.model_name,
        help="Base Model Architecture to use",
    )
    argparser.add_argument(
        "--num_epochs",
        type=int,
        default=default_cfg.num_epochs,
        help="number of training epochs",
    )
    argparser.add_argument(
        "--train_batch_size",
        type=int,
        default=default_cfg.train_batch_size,
        help="train batch size",
    )
    argparser.add_argument(
        "--eval_batch_size",
        type=int,
        default=default_cfg.eval_batch_size,
        help="eval batch size",
    )
    argparser.add_argument(
        "--warmup_steps",
        type=int,
        default=default_cfg.warmup_steps,
        help="warmup steps",
    )
    argparser.add_argument(
        "--learning_rate",
        type=float,
        default=default_cfg.learning_rate,
        help="learning rate to use",
    )
    argparser.add_argument(
        "--fp16",
        type=str,
        default=default_cfg.fp16,
        help="Whether to use floating point precision",
    )
    argparser.add_argument(
        "--log_model", action="store_true", help="log best model W&B"
    )
    return argparser.parse_args()


def compute_metrics(eval_pred):
    # define metrics and metrics function
    f1_metric = load_metric("f1")
    accuracy_metric = load_metric("accuracy")
    recall_metric = load_metric("recall")
    precision_metric = load_metric("precision")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # predictions.argmax(-1)
    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
        "recall": recall["recall"],
        "precision": precision["precision"],
    }


def load_data(run, cfg):

    # By including `use_artifact` we're logging the usage to W&B and can track it as part of the lineage
    train_artifact = run.use_artifact(f"{cfg.TRAIN_DATA_ARTIFACT}:latest")
    _ = train_artifact.download(root=cfg.TRAIN_DATA_FOLDER)
    train_dataset = load_from_disk(cfg.TRAIN_DATA_FOLDER)

    test_dataset = run.use_artifact(f"{cfg.TEST_DATA_ARTIFACT}:latest")
    _ = test_dataset.download(root=cfg.TEST_DATA_FOLDER)
    test_dataset = load_from_disk(cfg.TEST_DATA_FOLDER)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # tokenizer helper function
    def _tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    # tokenize dataset
    train_dataset = train_dataset.map(_tokenize, batched=True)
    test_dataset = test_dataset.map(_tokenize, batched=True)

    # set format for pytorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return train_dataset, test_dataset


def train(cfg):

    with wandb.init(
        project=cfg.PROJECT_NAME, job_type=cfg.MODEL_TRAINING_JOB_TYPE, config=dict(cfg)
    ) as run:

        cfg = wandb.config

        training_args = TrainingArguments(
            output_dir=cfg.MODEL_DATA_FOLDER,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.train_batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            warmup_steps=cfg.warmup_steps,
            fp16=cfg.fp16,
            learning_rate=float(cfg.learning_rate),
            # logging & evaluation strategies
            logging_dir=f"{cfg.MODEL_DATA_FOLDER}/logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="wandb",
            # push to hub parameters
            push_to_hub=cfg.PUSH_TO_HUB,
            hub_strategy=cfg.HUB_STRATEGY,
            hub_model_id=f"{cfg.model_name}-{cfg.PROJECT_NAME}-{cfg.warmup_steps}",
        )

        train_dataset, test_dataset = load_data(run, cfg)

        # Used to encode the labels
        label2id = train_dataset.features["labels"].str2int

        # Used later to initialise the model
        number_classes = train_dataset.features["labels"].num_classes
        id2label = train_dataset.features["labels"].int2str

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        
        # define data_collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            problem_type="single_label_classification",
            num_labels=number_classes,
        )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            compute_metrics=compute_metrics,
        )

        trainer.train()

        if cfg.log_model:
            trainer.save_model()

# def get_champion_metrics(cfg):
#     champion_model = use_model("model-registry/sub-product-classification").model_obj()
#     champion_model.load_state_dict(torch.load(cfg.champion_model_path))
#     champion_model.eval()
#     return champion_model


if __name__ == "__main__":
    default_cfg.update(vars(parse_args()))
    train(default_cfg)
