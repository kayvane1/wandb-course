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
from huggingface_hub import HfApi
import os
import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# defaults
default_cfg = config_dict.ConfigDict()

# WANDB BASE PARAMETERS
default_cfg.PROJECT_NAME = "wandb-week-3-complaints-classifier"
# WANDB ARTIFACT TYPES
default_cfg.DATASET_TYPE = "dataset"
default_cfg.MODEL_TYPE = "model"
# WANDB JOB TYPES
default_cfg.MODEL_EVALUATION_JOB_TYPE = "model-evaluation"
# WANDB ARTIFACT NAMES
default_cfg.VAL_DATA_ARTIFACT = "complaints_val_data"
# DATA FOLDERS
default_cfg.VAL_DATA_FOLDER = "complaints-dataset/val"
default_cfg.MODEL_DATA_FOLDER = "complaints-model"
# TRANSFORMERS PARAMETERS
default_cfg.model_id = "model-registry/complaints-product:champion"

def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(
        description="Process base parameters & hyper-parameters"
    )
    argparser.add_argument(
        "--model_name",
        type=str,
        default=default_cfg.model_name,
        help="Weights & Biases Alias to retrieve",
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
        predictions=predictions, references=labels, average="weighted"
    )
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(
        predictions=predictions, references=labels, average="weighted"
    )

    return {
        "accuracy": acc["accuracy"],
        "f1": f1["f1"],
        "recall": recall["recall"],
        "precision": precision["precision"],
    }


def load_data(run, cfg):

    # By including `use_artifact` we're logging the usage to W&B and can track it as part of the lineage
    val_artifact = run.use_artifact(f"{cfg.VAL_DATA_ARTIFACT}:latest")
    _ = val_artifact.download(root=cfg.VAL_DATA_FOLDER)
    val_dataset = load_from_disk(cfg.VAL_DATA_FOLDER)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    # tokenizer helper function
    def _tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    # tokenize dataset
    val_dataset = val_dataset.map(_tokenize, batched=True)

    # set format for pytorch
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return val_dataset


def evaluate(cfg):

    with wandb.init(
        project=cfg.PROJECT_NAME, job_type=cfg.MODEL_EVALUATION_JOB_TYPE, config=dict(cfg)
    ) as run:

        cfg = wandb.config

        artifact = run.use_artifact(cfg.model_id, type='model')
        #artifact_dir = artifact.download()
        #model_path = Path(artifact_dir).absolute()/'model'
        
        producer_run = artifact.logged_by()
        cfg.model_name = producer_run.config['model_name']
        cfg.hub_id = producer_run.config['hub_id']
        
        wandb.config.update(cfg)
        
        training_args = TrainingArguments(
            output_dir=cfg.MODEL_DATA_FOLDER,
            per_device_eval_batch_size=32,
            fp16=True,
            # logging & evaluation strategies
            logging_dir=f"{cfg.MODEL_DATA_FOLDER}/logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=1500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="wandb",
        )

        val_dataset = load_data(run, cfg)

        # Used to encode the labels
        label2id = val_dataset.features["labels"].str2int

        # Used later to initialise the model
        number_classes = val_dataset.features["labels"].num_classes
        id2label = val_dataset.features["labels"].int2str

        tokenizer = AutoTokenizer.from_pretrained(cfg.hub_id)
        
        # define data_collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.hub_id,
            problem_type="single_label_classification",
            num_labels=number_classes,
        )

        trainer = Trainer(
            model,
            training_args,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        outputs = trainer.evaluate()

        outputs_df = pd.DataFrame(data=outputs)
        run.log({f"{cfg.hub_id}-performance": wandb.Table(dataframe=outputs_df)})

if __name__ == "__main__":
    default_cfg.update(vars(parse_args()))
    evaluate(default_cfg)
