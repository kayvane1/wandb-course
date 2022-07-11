import argparse
from pathlib import Path
from datasets import load_dataset, load_from_disk
import numpy as np
import pandas as pd
import wandb
from ml_collections import config_dict
from transformers import AutoTokenizer

# defaults
default_cfg = config_dict.ConfigDict()

# WANDB BASE PARAMETERS
default_cfg.PROJECT_NAME = "wandb-week-2-complaints-classifier"
# WANDB ARTIFACT TYPES
default_cfg.DATASET_TYPE = "dataset"
default_cfg.MODEL_TYPE = "model"
# WANDB JOB TYPES
default_cfg.RAW_DATA_JOB_TYPE = "fetch-raw-data"
default_cfg.DATA_PROCESSING_JOB_TYPE = "process-data"
default_cfg.SPLIT_DATA_JOB_TYPE = "split-data"
# WANDB ARTIFACT NAMES
default_cfg.RAW_DATA_ARTIFACT = "complaints_raw_data"
default_cfg.PROCESSED_DATA_ARTIFACT = "complaints_processed_data"
default_cfg.TRAIN_DATA_ARTIFACT = "complaints_train_data"
default_cfg.TEST_DATA_ARTIFACT = "complaints_test_data"
# DATA FOLDERS
default_cfg.RAW_DATA_FOLDER = "complaints-dataset/raw"
default_cfg.PROCESSED_DATA_FOLDER = "complaints-dataset/processed"
default_cfg.TRAIN_DATA_FOLDER = "complaints-dataset/train"
default_cfg.TEST_DATA_FOLDER = "complaints-dataset/test"
# DATASET COLUMNS TO KEEP
default_cfg.dataset = "consumer-finance-complaints"
default_cfg.text_column = "Complaint Text"
default_cfg.target_column = "Sub Product"
default_cfg.split_perc = 10
# TRANSFORMERS PARAMETERS
default_cfg.model_name = "distilbert-base-uncased"


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(
        description="Prepare Data for Complaints Allocation"
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default=default_cfg.dataset,
        help="Name of the Dataset on the Huggingface Hub",
    )
    argparser.add_argument(
        "--text_column",
        type=str,
        default=default_cfg.text_column,
        help="Column in the dataset whichc ontains the text data / complaint background and narrative",
    )
    argparser.add_argument(
        "--target_column",
        type=str,
        default=default_cfg.target_column,
        help="Column in the dataset which should be used to as the target for predictions",
    )
    argparser.add_argument(
        "--split_perc",
        type=int,
        default=default_cfg.split_perc,
        help="percentage of the dataset to used",
    )
    return argparser.parse_args()


def log_raw_data(cfg):
    """
    Pulls the raw data from The Huggingface hub and logs it as an artifact.

    Args
        cfg (ConfigDict): ConfigDict object containing the configuration for the pipeline

    Returns:
        None

    """

    with wandb.init(
        project=cfg.PROJECT_NAME, job_type=cfg.RAW_DATA_JOB_TYPE, config=dict(cfg)
    ) as run:

        cfg = wandb.config

        if cfg.split_perc is not None:
            split_percentage = f"train[:{cfg.split_perc}%]"

        # Loading consumer complaints dataset - Note: This is a big dataset
        text_dataset = load_dataset(
            default_cfg.dataset, split=split_percentage, ignore_verifications=True
        )
        text_dataset.save_to_disk(cfg.RAW_DATA_FOLDER)
        # Create and log the raw data artifact
        raw_data_art = wandb.Artifact(cfg.RAW_DATA_ARTIFACT, type=cfg.DATASET_TYPE)
        raw_data_art.add_dir(cfg.RAW_DATA_FOLDER)
        run.log_artifact(raw_data_art)


def process_and_log_data(cfg):
    """
    Pulls the raw data from W&B and processes it into a dataset for training and testing

    Args:
        cfg (ConfigDict): ConfigDict object containing the configuration for the pipeline

    Returns:
        None
    """

    with wandb.init(
        project=cfg.PROJECT_NAME,
        job_type=cfg.DATA_PROCESSING_JOB_TYPE,
        config=dict(cfg),
    ) as run:

        cfg = wandb.config

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        # By including `use_artifact` we're logging the usage to W&B and can track it as part of the lineage
        text_artifact = run.use_artifact(f"{cfg.RAW_DATA_ARTIFACT}:latest")
        _ = text_artifact.download(root=cfg.RAW_DATA_FOLDER)
        text_dataset = load_from_disk(cfg.RAW_DATA_FOLDER)

        # Extracting the target column, there is only one split at this point (train)
        columns = text_dataset.column_names

        # Remove the columns which aren't in scope for us
        remove_cols = [
            e for e in columns if e not in (cfg.text_column, cfg.target_column)
        ]
        processed_data = text_dataset.remove_columns(remove_cols)

        # Renaming the columns to the names expected by the classifier
        processed_data = processed_data.rename_column(cfg.text_column, "text")
        processed_data = processed_data.rename_column(cfg.target_column, "labels")

        # Filtering out empty/no-text complaints
        processed_data = processed_data.filter(lambda example: len(example["text"]) > 0)

        processed_data.save_to_disk(cfg.PROCESSED_DATA_FOLDER)
        # Create and log the raw data artifact
        processed_data_art = wandb.Artifact(
            cfg.PROCESSED_DATA_ARTIFACT, type=cfg.DATASET_TYPE
        )
        processed_data_art.add_dir(cfg.PROCESSED_DATA_FOLDER)
        run.log_artifact(processed_data_art)


def split_and_log_data(cfg):
    """
    Splits the processed data into training and testing data, and logs it as artifacts.

    Args:
        cfg (ConfigDict): ConfigDict object containing the configuration for the pipeline.

    Returns:
        None

    """

    with wandb.init(
        project=cfg.PROJECT_NAME, job_type=cfg.SPLIT_DATA_JOB_TYPE, config=dict(cfg)
    ) as run:

        cfg = wandb.config

        # By including `use_artifact` we're logging the usage to W&B and can track it as part of the lineage
        processed_artifact = run.use_artifact(f"{cfg.PROCESSED_DATA_ARTIFACT}:latest")
        _ = processed_artifact.download(root=cfg.PROCESSED_DATA_FOLDER)
        processed_data = load_from_disk(cfg.PROCESSED_DATA_FOLDER)

        # Splitting the dataset into training and validation datasets
        split_data = processed_data.train_test_split(test_size=0.2, seed=0)

        train_dataset = split_data["train"]
        train_dataset.save_to_disk(cfg.TRAIN_DATA_FOLDER)

        test_dataset = split_data["test"]
        test_dataset.save_to_disk(cfg.TEST_DATA_FOLDER)

        # Create and log the train data artifact
        train_data_art = wandb.Artifact(cfg.TRAIN_DATA_ARTIFACT, type=cfg.DATASET_TYPE)
        train_data_art.add_dir(cfg.TRAIN_DATA_FOLDER)
        run.log_artifact(train_data_art)

        # Create and log the test data artifact
        test_data_art = wandb.Artifact(cfg.TEST_DATA_ARTIFACT, type=cfg.DATASET_TYPE)
        test_data_art.add_dir(cfg.TEST_DATA_FOLDER)
        run.log_artifact(test_data_art)


def run_data_pipeline(cfg):
    """
    Runs the data processing pipeline.
    """

    log_raw_data(cfg)
    process_and_log_data(cfg)
    split_and_log_data(cfg)


if __name__ == "__main__":
    default_cfg.update(vars(parse_args()))
    run_data_pipeline(default_cfg)
