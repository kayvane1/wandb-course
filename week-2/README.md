# Week 2 - Weights & Biases ML Ops Course

## Assets & Links

| Assignment      | Colab | W&B Report | Description |
| ----------- | ----------- | -------| ------|
| Week 2 - Complaints Allocation Hyperparameter Sweep   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mvUHh_uLOS7TZBUjfXjr_hPCBxvLg-g-?usp=sharing)     | [![Visualise in W&B](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/kayvane/wandb-week-2-complaints-classifier/reports/Complaints-Allocation-with-Sweeps--VmlldzoyMzI5MTQw?accessToken=y5gvb2af7vb9bscetht3c02j5af34q5z65vifnfpshhd4j5uaqbdh5y33vjxfxys)| The notebook is split into python files for maintainability and parameterised to allow the team to easily experiment with multiple models available on the huggingface hub as well as introducing hyperparameter optimisation using W&B Sweeps|

## Improvements:

The notebook has been split into two python scripts `process_data.py` and `train.py`, some of the arguments can be overidden through the command line to enable W&B sweeps to be run. Additionally, the `text_column` and `target_column` have been parameterised as we know we will need to eventually train more than one model on this dataset as the business wants to also use the system to identify customers who are likely to be vulnerable using the `XXXX` column

To run the data procssing step, overriding any default arguments, we use the following command:

```
!python process_data.py --split_perc 10
```

Tokenisation has been removed from the data processing pipeline and included as part of the training script. While this increases overhead time it allows the team to experiment with multiple model architectures which require different tokenization techniques. This is achieved using HuggingFace's `Automodel` classes.

```python

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    cfg.model_name,
    problem_type="single_label_classification",
    num_labels=number_classes,
    id2label=id2label,
    label2id=label2id
)

```

# Sweeps

By parameterising the training script, we can now run hyperparameter optimisation using weights and biases sweeps.

This is performed by running two simple commands.

First, we instantiate a sweep using the following command:

```
!wandb sweep sweep.yaml
#[OUT] wandb: Creating sweep from: sweep.yaml
#[OUT] wandb: Created sweep with ID: hvvwvj4n
#[OUT] wandb: View sweep at: https://wandb.ai/kayvane/wandb-week-2-complaints-classifier/sweeps/hvvwvj4n
#[OUT] wandb: Run sweep agent with: wandb agent kayvane/wandb-week-2-complaints-classifier/hvvwvj4n
```

Next we use the sweep agent's id to start a run: 

```
!wandb agent kayvane/wandb-week-2-complaints-classifier/hvvwvj4n --count 10
```

We need to set the number of runs we want to trial with the --count arg, as our training takes about 1h if we select 10% of the data, we'll set 10 runs with the expectation the sweep will take 10h to complete.

# Sweeps config

In this sweep we are looking to maximize f1, but parameterising the sweep in this way also allows us to change the optimisation metric with ease and evaluate/compare the model's performance seperately in a W&B Report.

We also use the `early_terminate` option which uses the `hyperband` algorithm

```
program: train.py
method: random
project: wandb-week-2-complaints-classifier
metric:
  name: eval/f1
  goal: maximize
early_terminate:
  type: hyperband
  min_iter: 3
parameters:
  train_batch_size:
    value: 16
  eval_batch_size:
    value: 16
  warmup_steps:
    values: [256, 512, 1024]
  model_name:
    values:
      - 'distilbert-base-uncased'
      - 'distilroberta-base'
  learning_rate:
    distribution: 'log_uniform_values'
    min: 1e-5
    max: 1e-3
```

References:
- W&B ML Ops course: https://github.com/wandb/edu
- https://docs.wandb.ai/guides/integrations/huggingface
- https://docs.wandb.ai/guides/track
- https://docs.wandb.ai/guides/sweeps