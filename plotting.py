import matplotlib.pyplot as plt

def plot_training_results(dates, actual, predicted, title="Model Predictions"):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual Target', color='black')
    plt.plot(dates, predicted, label='Predicted', color='blue', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
