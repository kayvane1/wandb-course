import argparse
from torch import tensor, nn, device, cuda
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers.trainer_callback import EarlyStoppingCallback
from huggingface_hub import HfFolder
from datasets import load_metric
import numpy as np
import pandas as pd
from ml_collections import config_dict
import wandb


# defaults
default_cfg = config_dict.ConfigDict()

# WANDB BASE PARAMETERS
default_cfg.PROJECT_NAME = 'wandb-week-2-complaints-classifier'
# WANDB ARTIFACT TYPES
default_cfg.DATASET_TYPE = "dataset"
default_cfg.MODEL_TYPE = "model"
# WANDB JOB TYPES
default_cfg.MODEL_TRAINING_JOB_TYPE = "model-training"
# WANDB ARTIFACT NAMES
default_cfg.TRAIN_DATA_ARTIFACT = "complaints_train_data"
default_cfg.TEST_DATA_ARTIFACT = "complaints_test_data"
# DATA FOLDERS
default_cfg.TRAIN_DATA_FOLDER = 'complaints-dataset/train'
default_cfg.TEST_DATA_FOLDER = 'complaints-dataset/test'
default_cfg.MODEL_DATA_FOLDER = 'complaints-model'
# TRANSFORMERS PARAMETERS
default_cfg.model_name = "distilbert-base-uncased"
default_cfg.num_epochs = 3
default_cfg.train_batch_size = 32
default_cfg.eval_batch_size = 32
default_cfg.warmup_steps = 500
default_cfg.learning_rate = 5e-5
default_cfg.fp16 = True
# HUB PARAMETERS
default_cfg.PUSH_TO_HUB = True
default_cfg.HUB_STRATEGY = "every_save"


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Process base parameters & hyper-parameters')
    argparser.add_argument('--model_name', type=int, default=default_cfg.model_name, help='Base Model Architecture to use')
    argparser.add_argument('--num_epochs', type=int, default=default_cfg.num_epochs, help='number of training epochs')
    argparser.add_argument('--train_batch_size', type=int, default=default_cfg.train_batch_size, help='train batch size')
    argparser.add_argument('--eval_batch_size', type=int, default=default_cfg.eval_batch_size, help='eval batch size')
    argparser.add_argument('--warmup_steps', type=float, default=default_cfg.warmup_steps, help='warmup steps')
    argparser.add_argument('--learning_rate', type=str, default=default_cfg.learning_rates, help='learning rate to use')
    argparser.add_argument('--fp16', type=str, default=default_cfg.fp16, help='Whether to use floating point precision')
    argparser.add_argument('--log_model', action="store_true", help='log best model W&B')
    return argparser.parse_args()


def compute_metrics(eval_pred):
  # define metrics and metrics function
  f1_metric = load_metric("f1")
  accuracy_metric = load_metric( "accuracy")
  recall_metric = load_metric("recall")
  precision_metric = load_metric("precision")
  
  predictions, labels = eval_pred
  predictions = np.argmax(predictions, axis=1) # predictions.argmax(-1)
  acc = accuracy_metric.compute(predictions=predictions, references=labels)
  recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')
  f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
  precision = precision_metric.compute(predictions=predictions, references=labels, average="macro")

  return {
      "accuracy": acc["accuracy"],
      "f1": f1["f1"],
      "recall": recall["recall"],
      "precision" : precision["precision"]
  }

def train(cfg):

    with wandb.init(project=cfg.PROJECT_NAME, job_type=cfg.MODEL_TRAINING_JOB_TYPE, config=dict(cfg)) as run:

        cfg = wandb.config
    
        training_args = TrainingArguments(
        output_dir=cfg.MODEL_DATA_FOLDER,
        num_train_epochs=cfg.NUM_EPOCHS,
        per_device_train_batch_size=cfg.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=cfg.EVAL_BATCH_SIZE,
        warmup_steps=cfg.WARMUP_STEPS,
        fp16=cfg.FP16,
        learning_rate=float(cfg.LEARNING_RATE),
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
        hub_model_id=f"{cfg.model_name}-{cfg.PROJECT_NAME}",
        )

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        # By including `use_artifact` we're logging the usage to W&B and can track it as part of the lineage
        train_dataset = run.use_artifact(f'{cfg.TRAIN_DATA_ARTIFACT}:latest')
        test_dataset = run.use_artifact(f'{cfg.TEST_DATA_ARTIFACT}:latest')

        # set format for pytorch
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        # Used to encode the labels
        label2id = train_dataset.features["label"].str2int

        # Used later to initialise the model
        number_classes = train_dataset.features["label"].num_classes
        id2label = train_dataset.features["label"].int2str

        # define data_collator
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=number_classes
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