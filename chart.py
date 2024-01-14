import matplotlib.pyplot as plt

data = [
    {
        "faithfulness": 1.0,
        "answer_relevancy": 0.8792,
        "context_precision": 1.0,
        "context_recall": 1.0,
    },
    {
        "faithfulness": 1.0,
        "answer_relevancy": 0.989,
        "context_precision": 0.0,
        "context_recall": 1.0,
    },
    {
        "faithfulness": 1.0,
        "answer_relevancy": 0.9861,
        "context_precision": 1.0,
        "context_recall": 1.0,
    },
    {
        "faithfulness": 1.0,
        "answer_relevancy": 0.996,
        "context_precision": 1.0,
        "context_recall": 1.0,
    },
    {
        "faithfulness": 0.0,
        "answer_relevancy": 0.9803,
        "context_precision": 0.0,
        "context_recall": 1.0,
    },
    {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 1.0,
    },
]

# Extract data for each metric
faithfulness = [datum["faithfulness"] for datum in data]
answer_relevancy = [datum["answer_relevancy"] for datum in data]
context_precision = [datum["context_precision"] for datum in data]
context_recall = [datum["context_recall"] for datum in data]

# Create a line chart for each metric
plt.figure(figsize=(10, 6))

plt.plot(faithfulness, label="Faithfulness")
plt.plot(answer_relevancy, label="Answer Relevancy")
plt.plot(context_precision, label="Context Precision")
plt.plot(context_recall, label="Context Recall")

# Add labels and title
plt.xlabel("Data Point")
plt.ylabel("Metric Value")
plt.title("Model Evaluation Metrics")

# Add legend
plt.legend()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.grid(True)
plt.tight_layout()
plt.show()