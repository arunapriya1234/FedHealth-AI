# FedHealth AI 

## Privacy-Preserving Federated Learning for Wearable Human Activity Recognition

FedHealth AI is a machine learning project that explores **Federated Learning (FL)** and **Differential Privacy (DP)** for human activity recognition using wearable sensor data.

##  Objectives

- Implement federated learning using Federated Averaging (FedAvg).
- Explore differential privacy during local model training.
- Classify six human activities using wearable sensor features.
- Evaluate model performance and privacy trade-offs.

##  Dataset

The project uses the **UCI Human Activity Recognition Using Smartphones dataset**.

- **Input features:** 561
- **Activity classes:** 6
- **Federated clients:** 5

### Activity Classes

| Label | Activity |
|---:|---|
| 1 | Walking |
| 2 | Walking Upstairs |
| 3 | Walking Downstairs |
| 4 | Sitting |
| 5 | Standing |
| 6 | Laying |

##  Model Architecture

The project uses a Multilayer Perceptron (MLP).

| Component | Configuration |
|---|---|
| Input layer | 561 features |
| Hidden layer 1 | 128 neurons |
| Hidden layer 2 | 64 neurons |
| Output layer | 6 classes |
| Activation | ReLU |

## Experimental Results

| Experiment | Test Accuracy | Macro F1-score |
|---|---:|---:|
| Non-private FedAvg baseline | 93.25% | 93.19% |
| Differential privacy, 5 local epochs | 76.04% | 70.05% |
| Differential privacy with Sitting class weighting | 76.69% | 71.26% |
| Sitting decision-threshold adjustment | **83.31%** | **81.96%** |

These results are specific to the experiments conducted. The threshold adjustment is a post-processing step, not a separately trained model.

##  Technologies

- Python
- PyTorch
- Flower
- Opacus
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

##  Project Structure

```text
FedHealth-AI/
├── src/                  # Source code and training scripts
├── tests/                # Automated tests
├── data/                 # Local datasets (excluded from Git)
├── experiments/          # Experiment outputs (excluded from Git)
├── results/              # Generated reports (excluded from Git)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── .gitignore            # Git ignore rules
```

##  Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/arunapriya1234/FedHealth-AI.git
cd FedHealth-AI
```

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the tests

```powershell
python -m pytest tests/ -v
```

Ensure the required dataset is available in the expected local location before running the data tests or training scripts.

##  Privacy Considerations

This project experimentally explores federated learning and differential privacy.

- Federated learning supports collaborative training without centrally combining the original training samples in the basic workflow.
- Differential privacy mechanisms are intended to limit the influence of individual training examples.
- Privacy mechanisms can affect model performance.
- Privacy-accounting estimates depend on the training configuration and assumptions.
- The reported privacy accounting has not been independently verified as a complete end-to-end privacy guarantee.

**This project is experimental and is not a production-ready healthcare system.** The reported results should not be interpreted as clinical validation or proof of a complete privacy guarantee.

## 🔮 Future Improvements

- Improve model performance under differential privacy.
- Analyze confusion matrices and per-class performance.
- Evaluate additional privacy configurations.
- Verify privacy accounting and secure random number generation.
- Investigate the impact of different client data distributions.
- Improve experiment reproducibility and documentation.

##  Project Purpose

FedHealth AI explores the intersection of wearable human activity recognition, federated learning, and differential privacy.

---

**Disclaimer:** The experimental metrics reflect the configurations and evaluation procedures used in this project. Further testing is needed before real-world deployment.