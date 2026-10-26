
from pathlib import Path
from datetime import date

import pandas as pd


# ============================================================
# FEDHEALTH AI - FINAL EXPERIMENT REPORT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_PATH = RESULTS_DIR / "FedHealth_AI_Final_Experiment_Report.md"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def pct(value):
    return f"{value:.2f}%"


def load_csv_if_exists(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as exc:
            print(f"Could not read {path.name}: {exc}")
    return None


def main():

    report = []

    def add(text=""):
        report.append(text)

    add("# FedHealth AI")
    add("## Final Experiment Comparison Report")
    add()
    add(f"**Report generated:** {date.today().isoformat()}")
    add("**Project:** Privacy-Preserving Federated Learning for Wearable Human Activity Recognition")
    add("**Framework:** Python, PyTorch, Opacus, and scikit-learn")
    add()
    add("---")
    add()

    # ========================================================
    # 1. ABSTRACT
    # ========================================================

    add("## 1. Abstract")
    add()
    add(
        "FedHealth AI investigates federated learning for wearable "
        "human activity recognition while incorporating differential "
        "privacy during client-side model training. The experiments use "
        "the UCI Human Activity Recognition (HAR) dataset, containing "
        "561 input features and six activity classes."
    )
    add()
    add(
        "The project compares a non-private FedAvg baseline with "
        "differentially private federated models. Further experiments "
        "investigate class-weighted training and a post-processing "
        "decision adjustment designed to improve recognition of Sitting "
        "activity, which was frequently confused with Standing."
    )
    add()
    add(
        "The best recorded experiment achieved 83.31% official test "
        "accuracy and 81.96% macro-F1 after applying a Sitting decision "
        "multiplier selected using validation data to a differentially "
        "private Sitting-Boost model. This adjustment improves predictive "
        "performance but does not itself establish a differential privacy "
        "guarantee."
    )
    add()

    # ========================================================
    # 2. OBJECTIVES
    # ========================================================

    add("## 2. Project Objectives")
    add()
    add("1. Build a federated human activity recognition system.")
    add("2. Distribute training data across five simulated clients.")
    add("3. Apply differential privacy during local model training.")
    add("4. Compare private and non-private federated learning.")
    add("5. Investigate activity-level classification errors.")
    add("6. Improve Sitting recognition without retraining the final model.")
    add("7. document model performance and privacy-accounting limitations.")
    add()

    # ========================================================
    # 3. DATASET
    # ========================================================

    add("## 3. Dataset and Activity Classes")
    add()
    add(
        "The project uses the UCI HAR dataset. The model receives "
        "561 engineered features for each observation and predicts "
        "one of six activities."
    )
    add()
    add("| Model label | Activity |")
    add("|---:|---|")
    activities = [
        "Walking",
        "Walking Upstairs",
        "Walking Downstairs",
        "Sitting",
        "Standing",
        "Laying",
    ]
    for index, activity in enumerate(activities):
        add(f"| {index} | {activity} |")
    add()
    add(
        "The original dataset activity labels range from 1 to 6. "
        "The implementation converts these labels to model indices "
        "0 to 5 by subtracting one."
    )
    add()
    add(
        "The recorded official test evaluation contains 2,947 samples. "
        "The model uses 561 input features and six output classes."
    )
    add()

    # ========================================================
    # 4. SYSTEM ARCHITECTURE
    # ========================================================

    add("## 4. System Architecture")
    add()
    add("The recorded training configuration uses five clients.")
    add()
    add("```text")
    add("UCI HAR Dataset")
    add("       |")
    add("       v")
    add("Feature preparation and client data partitioning")
    add("       |")
    add("       v")
    add("Five simulated federated clients")
    add("       |")
    add("       +--> Local model training")
    add("       |      +--> Cross-entropy loss")
    add("       |      +--> Adam optimizer")
    add("       |      +--> Opacus differential privacy")
    add("       |")
    add("       v")
    add("Server-side Federated Averaging (FedAvg)")
    add("       |")
    add("       v")
    add("Global model evaluation")
    add("       |")
    add("       v")
    add("Optional Sitting-vs-Standing decision adjustment")
    add("       |")
    add("       v")
    add("Predicted human activity")
    add("```")
    add()
    add(
        "This is a simulated federated setup. The experiments do not "
        "by themselves demonstrate deployment across physically "
        "separate wearable devices."
    )
    add()

    # ========================================================
    # 5. MODEL AND CONFIGURATION
    # ========================================================

    add("## 5. Model and Experimental Configuration")
    add()
    add("The recorded MLP architecture is:")
    add()
    add("```text")
    add("Input layer: 561 features")
    add("Hidden layer 1: 128 units, ReLU")
    add("Hidden layer 2: 64 units, ReLU")
    add("Output layer: 6 activity classes")
    add("```")
    add()
    add("| Setting | Recorded value |")
    add("|---|---|")
    add("| Number of clients | 5 |")
    add("| Communication rounds | 10 |")
    add("| Local epochs in the main DP experiments | 2 or 5 |")
    add("| Batch size | 32 |")
    add("| Learning rate | 0.001 |")
    add("| DP noise multiplier in the main comparison | 0.7 |")
    add("| Maximum gradient norm | 2.0 |")
    add("| Device | CPU |")
    add()
    add(
        "The main private experiments use Opacus for gradient clipping "
        "and noise addition. The recorded accountant diagnostics use "
        "delta = 1e-6 per client. Their interpretation as an overall "
        "federated privacy guarantee requires further verification."
    )
    add()

    # ========================================================
    # 6. EXPERIMENT COMPARISON
    # ========================================================

    add("## 6. Main Experiment Results")
    add()
    add(
        "The following results were recorded during the experiments. "
        "Previously recorded metrics that were not recalculated in "
        "the current evaluation scripts are identified separately."
    )
    add()
    add("| Model / experiment | Test accuracy | Macro-F1 | Weighted-F1 |")
    add("|---|---:|---:|---:|")
    add(
        "| Non-private FedAvg, no dropout "
        "| 93.25% | 93.19% | 93.25% |"
    )
    add(
        "| DP-FedAvg, 2 local epochs, clip 2.0 "
        "| 69.73% | 63.08% | 63.56% |"
    )
    add(
        "| DP-FedAvg, 5 local epochs "
        "| 76.04% | 70.05% | 70.35% |"
    )
    add(
        "| DP-FedAvg with class-weighted training "
        "| 76.15% | 70.22% | 70.51% |"
    )
    add(
        "| DP-FedAvg with Sitting-Boost training "
        "| 76.69% | 71.26% | 71.55% |"
    )
    add(
        "| Sitting-Boost with validation-selected x4 adjustment "
        "| **83.31%** | **81.96%** | **82.38%** |"
    )
    add()
    add(
        "**Important:** The non-private FedAvg baseline is a useful "
        "performance reference, not a privacy-equivalent comparison. "
        "The x4 result uses the same trained Sitting-Boost model with "
        "adjusted predictions, not an independently retrained model."
    )
    add()

    # ========================================================
    # 7. SITTING BOOST
    # ========================================================

    add("## 7. Sitting Recognition Investigation")
    add()
    add(
        "The original Sitting-Boost model continued to confuse Sitting "
        "with Standing. On the official test set, the original model "
        "correctly identified 24 of 491 actual Sitting samples."
    )
    add()
    add("| Sitting metric | Original model | With x4 adjustment |")
    add("|---|---:|---:|")
    add("| Correct Sitting predictions | 24 | 241 |")
    add("| Sitting precision | 100.00% | 91.63% |")
    add("| Sitting recall | 4.89% | 49.08% |")
    add("| Sitting F1 | 9.32% | 63.93% |")
    add()
    add(
        "The x4 decision adjustment multiplies the Sitting probability "
        "by four when comparing it with the Standing probability. "
        "When the adjusted Sitting probability exceeds the Standing "
        "probability, an original Standing prediction is changed to "
        "Sitting. Other original predictions remain unchanged."
    )
    add()
    add(
        "The multiplier was selected by comparing candidate settings "
        "on validation data, with macro-F1 as the selection criterion "
        "and accuracy used to break ties. The fixed setting was then "
        "evaluated on the official test set."
    )
    add()
    add(
        "This is a validation-selected decision rule. It does not "
        "modify the trained model parameters, and its performance "
        "should not be assumed to generalize identically to other "
        "datasets or deployment environments."
    )
    add()

    # ========================================================
    # 8. PRIVACY ACCOUNTING
    # ========================================================

    add("## 8. Differential Privacy and Accounting")
    add()
    add(
        "The private client-training code uses Opacus with gradient "
        "clipping and Gaussian noise. The recorded five-epoch "
        "Sitting-Boost run reports a per-client accountant diagnostic "
        "of epsilon = 20.8534 at delta = 1e-6."
    )
    add()
    add(
        "This value must not be presented as a verified end-to-end "
        "privacy guarantee for the complete federated system without "
        "checking the accounting assumptions and composition across "
        "clients and communication rounds."
    )
    add()
    add(
        "The training logs also indicate that secure random number "
        "generation was disabled. These experiments should therefore "
        "be treated as experimental results, not as a production-ready "
        "privacy implementation."
    )
    add()
    add(
        "The prediction adjustment is post-processing of the model's "
        "output. It does not add another training step, but the overall "
        "privacy claim still depends on the validity of the training "
        "mechanism, accounting, data handling, and release procedure."
    )
    add()

    # ========================================================
    # 9. LIMITATIONS
    # ========================================================

    add("## 9. Limitations")
    add()
    add("1. The experiments use a single benchmark dataset.")
    add(
        "2. Five clients are simulated; real-device communication "
        "and deployment have not been established."
    )
    add(
        "3. The Sitting-versus-Standing decision adjustment improves "
        "recognition but introduces additional false Sitting predictions."
    )
    add(
        "4. The reported per-client privacy diagnostics need an "
        "end-to-end accounting review."
    )
    add(
        "5. Secure random number generation was disabled during "
        "the recorded experiments."
    )
    add(
        "6. Further repeated runs with fixed seeds and uncertainty "
        "estimates would strengthen the evidence."
    )
    add(
        "7. Additional datasets and client distributions are needed "
        "to evaluate generalization."
    )
    add(
        "8. The model has not been established as a clinically or "
        "safety-critical activity recognition system."
    )
    add()

    # ========================================================
    # 10. CONCLUSION
    # ========================================================

    add("## 10. Conclusion")
    add()
    add(
        "FedHealth AI demonstrates an experimental federated learning "
        "pipeline for wearable human activity recognition. The recorded "
        "results show a difference between the non-private baseline "
        "and the differentially private models, highlighting the "
        "importance of studying both predictive utility and privacy."
    )
    add()
    add(
        "The main activity-level challenge was the confusion between "
        "Sitting and Standing. A validation-selected x4 Sitting "
        "decision adjustment increased official test accuracy from "
        "76.69% to 83.31% and macro-F1 from 71.26% to 81.96% for the "
        "Sitting-Boost model."
    )
    add()
    add(
        "The results are promising but do not eliminate the need for "
        "a rigorous end-to-end privacy audit, secure randomness, "
        "additional validation, and evaluation on other datasets."
    )
    add()

    # ========================================================
    # 11. REPRODUCIBILITY FILES
    # ========================================================

    add("## 11. Reproducibility and Output Files")
    add()
    add("Important project files include:")
    add()
    add("| Purpose | Path |")
    add("|---|---|")
    add(
        "| MLP architecture | `src/models_no_dropout.py` |"
    )
    add(
        "| DP client training | `src/client_dp_balanced_no_dropout.py` |"
    )
    add(
        "| DP server training | `src/server_dp_balanced_clip2.py` |"
    )
    add(
        "| Sitting-Boost client | `src/client_dp_balanced_sittingboost.py` |"
    )
    add(
        "| Sitting-Boost server | `src/server_dp_balanced_sittingboost_epochs5.py` |"
    )
    add(
        "| Validation multiplier tuning | `src/tune_sitting_threshold.py` |"
    )
    add(
        "| Official test evaluation | `src/evaluate_dp_sittingboost_test.py` |"
    )
    add(
        "| Threshold-adjusted test results | "
        "`experiments/balanced_dp_sittingboost_threshold4_test_evaluation/` |"
    )
    add()
    add(
        "The recorded test comparison is stored in "
        "`test_model_comparison.csv` inside the threshold-adjusted "
        "evaluation directory."
    )
    add()

    # ========================================================
    # 12. INTERPRETATION NOTE
    # ========================================================

    add("## 12. Reporting Guidance")
    add()
    add(
        "When presenting this project, distinguish the measured "
        "classification results from claims about privacy guarantees. "
        "Use the phrase 'per-client accountant diagnostic' until the "
        "overall federated privacy accounting has been verified."
    )
    add()
    add(
        "Report the 83.31% accuracy as the result of the "
        "Sitting-Boost model with a validation-selected x4 decision "
        "adjustment, rather than describing it as the accuracy of "
        "the unadjusted model."
    )
    add()
    add(
        "Do not claim that the decision adjustment improves privacy. "
        "Its measured benefit is improved classification performance "
        "on the evaluated dataset."
    )
    add()

    # ========================================================
    # OPTIONAL: INCLUDE AVAILABLE GENERATED COMPARISON CSV
    # ========================================================

    comparison_csv = RESULTS_DIR / "detailed_model_comparison.csv"
    comparison_df = load_csv_if_exists(comparison_csv)

    if comparison_df is not None:
        add("## Appendix A. Existing Detailed Comparison Data")
        add()
        add(
            "The project also contains "
            "`results/detailed_model_comparison.csv`. "
            "Consult that CSV for the previously recorded model "
            "comparison details; its contents are not silently "
            "merged into the manually recorded table above."
        )
        add()

    REPORT_PATH.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("=" * 70)
    print("FEDHEALTH AI REPORT GENERATED")
    print("=" * 70)
    print(f"\nReport saved to:\n{REPORT_PATH}")
    print("\nThe report includes:")
    print("1. Abstract and objectives")
    print("2. Dataset and architecture")
    print("3. Experimental configurations")
    print("4. Recorded model comparisons")
    print("5. Sitting-Boost and x4 adjustment results")
    print("6. Privacy-accounting caveats")
    print("7. Limitations and conclusion")
    print("\nOpen the Markdown report in VS Code to review it.")


if __name__ == "__main__":
    main()
