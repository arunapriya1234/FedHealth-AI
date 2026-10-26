# FedHealth AI

## Privacy-Preserving Federated Learning for Wearable Human Activity Recognition



FedHealth AI is a machine learning project that explores federated learning and differential privacy for human activity recognition using wearable sensor data.



## Objectives

\- Implement federated learning using Federated Averaging (FedAvg).

\- Explore differential privacy during local model training.

\- Classify six human activities using wearable sensor features.

\- Evaluate model performance and privacy trade-offs.



## Dataset

The project uses the UCI Human Activity Recognition Using Smartphones dataset.



\- Input features: 561

\- Activity classes: 6

\- Federated clients: 5



### Activity Classes

1\. Walking

2\. Walking Upstairs

3\. Walking Downstairs

4\. Sitting

5\. Standing

6\. Laying



## Model

A multilayer perceptron (MLP) with:

\- Input layer: 561 features

\- Hidden layers: 128 and 64 neurons

\- Output layer: 6 classes

\- Activation: ReLU



## Experimental Results



| Experiment | Test Accuracy | Macro F1-score |

|---|---:|---:|

| Non-private FedAvg baseline | 93.25% | 93.19% |

| Differential privacy, 5 local epochs | 76.04% | 70.05% |

| Differential privacy with Sitting class weighting | 76.69% | 71.26% |

| Sitting decision-threshold adjustment | 83.31% | 81.96% |



These results are specific to the experiments conducted. The threshold adjustment is a post-processing step, not a separately trained model.



## Technologies

\- Python

\- PyTorch

\- Flower

\- Opacus

\- NumPy

\- Pandas

\- Scikit-learn

\- Matplotlib



## Project Structure

\- `src/` — Source code and training scripts

\- `tests/` — Tests

\- `data/` — Local datasets, excluded from Git

\- `experiments/` — Experiment outputs, excluded from Git

\- `results/` — Generated reports, excluded from Git



## Setup



Create a virtual environment:



```bash

python -m venv venv


