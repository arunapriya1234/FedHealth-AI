
import matplotlib.pyplot as plt

from data_processing import load_dataset


def plot_activity_distribution():
    _, y_train, _, _, _, activities = load_dataset()

    counts = (
        y_train["activity"]
        .value_counts()
        .sort_index()
    )

    labels = activities.set_index("activity")["activity_name"]
    activity_names = [labels.loc[i] for i in counts.index]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(activity_names, counts.values)

    plt.title("UCI-HAR Training Activity Distribution")
    plt.xlabel("Activity")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=25, ha="right")

    for bar, count in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_dir = __import__("pathlib").Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "activity_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Chart saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    plot_activity_distribution()
