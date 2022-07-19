<img src="https://i.imgur.com/gb6B4ig.png" width="400" alt="Weights & Biases" />

Assignements done for the Weights & Biases ML Ops course.

| Assignment      | Colab | W&B Report | Description |
| ----------- | ----------- | -------| ------|
| Week 1 - Complaints Allocation Base      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FKQcSFY9ShHPYzd--5L4aIebvmoRqKrm?usp=sharing)       |[![Visualise in W&B](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/kayvane/wandb-week-1-complaints-classification/reports/Complaints-Allocation--VmlldzoyMjYzNzM3)| A notebook is used to build the base pipeline integrating Weights and Biases with the HuggingFace ecosystem|
| Week 2 - Complaints Allocation Hyperparameter Sweep   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mvUHh_uLOS7TZBUjfXjr_hPCBxvLg-g-?usp=sharing)     | [![Visualise in W&B](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/kayvane/wandb-week-2-complaints-classifier/reports/Complaints-Allocation-with-Sweeps--VmlldzoyMzI5MTQw?accessToken=y5gvb2af7vb9bscetht3c02j5af34q5z65vifnfpshhd4j5uaqbdh5y33vjxfxys)| The notebook is split into python files for maintainability and parameterised to allow the team to easily experiment with multiple models available on the huggingface hub as well as introducing hyperparameter optimisation using W&B Sweeps|
| Week 3 - Model Management & Evaluation   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15PnwkMunSN-uc5gVI4J83WvA3cW6J0dH?usp=sharing)  --       |[![Visualise in W&B](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/kayvane/wandb-week-3-complaints-classifier/reports/Complaints-Allocation--VmlldzoyMzMyMjA4?accessToken=4gqfg8ueabt18jn2a9jklxkcel0dufmxib3f54fcyurufx366i92i019603iv3af) | An Evaluation script is added which allows for models to be trained over time to be evaluated against one another, additional changes are made to the way data is versionned to reduce the probability of data leakage, a simplified version of a Champion-Challenger model is introduced so the system can be run every week|


# Problem Description

The complaints department in the fictional _MegaBank_ recieves customer complaints and enquiries which need to be triaged to the right teams. In order to do this there is a triage team which screens the incoming complaints and routes them to the right product team. Each product team has it's own recurring issues it is aware of and has policies and procedures to resolve them in a systematic way, in addition to those, new issues may arise where a complaints analyst will need to use their best judgement to address the complaint. _MegaBank_ also has a commitment to the regulator to ensure vulnerable customers (the elderly / ex-service people) follow a different customer journey to ensure they are seen to by qualified analysts.  

# Business Value Analysis

The Triage team are a team of 8 Analysts with one supervisor, previous analysis has shown the Triage teams accuracy is 90% in correctly sending complaints to the right teams. However, as new issue types arise, these tend to get blocked until the correct routing is identified. It is not expected that the model will deal well with new issue types / low frequency issues as these require organisational insight, ultimately needing to be routed to a specialist team first to build the correct user journey for the problem type, then assigning it to the most relevant team and training the team to deal with the issue consistently.

The department head wants to redeploy the analysts to the specialist investigations team as it is under-staffed and the Triage team have a good understanding of a wide range of issues. They would require less upskilling than hiring new members to the team and know how the company operates. With an average analyst salary of £30k for analysts and £50k for the supervisor, the model could save the company up to £290k + cost savings of the recruitment process. 

The department head has been told that the model may perform worse on lower occurring issues, over time if the issues arise more often the model's performance will get better as a result of more data becoming available.


__Summary:__

Potential Cost Saving: 290k
Potential Employee redeployment: 9 FTE
Benchmark Accuracy: 90%
Known Issues (1): New Issue types require organisational design to plan and approve the correct customer journey
Accepted Caveats (1): Management have been informed that low volume issue types will have a lower performance and may be mis-routed.

## Functional Requirements

- The implementation must not change the current ways of working of the operations team
- The implementation must be designed with a feedback loop in place so it can adapt to new issues
- All outputs should be aligned to a specific model version to ensure all outcomes are traceable
- Decision explainability is not required at the time of allocation but should be available if requested by the regulator
- The system should be able to extend if required, e.g. including an additional model 'task' in the pipeline
- The system should be able to handle different model deployment strategies when a new model becomes available

## Non-Functional Requirements

- System should be able to scale between 3k and 15k complaints to be processed per day
- System should cut the time of allocation by at least 50% - it currently takes the triage team 1-3 minutes to triage an incoming complaint 
- A cost saving of at least 50% is expected on the price of the existing BAU team, even though they are being redeployed. This should include the cost the infrastructure, services, database usage etc.


# Implementation Details

Tools & Infrastructure:

**Modelling**: HuggingFace Transformers
**Hyperparameter Optimisation**: Weights & Biases
**Data Storage**: HuggingFace Hub  
**Model Storage**: HuggingFace Hub  
**Model Management**: Weights and Biases
**Experiment Tracking**: Weights and Biases  

The ML Team use the HuggingFace Hub to store to back up their dataset every day, Weights and Biases is used to track the lineage of the pre-processing pipeline, the model registry is used to keep track of the models through the model management lifecycle process.

# The Dataset

