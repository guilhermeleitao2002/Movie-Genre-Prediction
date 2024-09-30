#!/usr/bin/env python3

from subprocess import run
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend
import matplotlib.pyplot as plt
from time import time

# Define argument ranges for experimentation
max_features_values = [4000, 5000, 6000]
ngram_ranges = [(1, 2), (1, 3)]
combine_fields_values = ['from,director']

# Store results for plotting
nb_accuracies = []
svm_accuracies = []
experiment_labels = []

# Function to run base.py and capture output accuracies
def run_experiment(max_features, ngram_range, combine_fields):
    # Run base.py with different arguments
    ngram_range_str = f"{ngram_range[0]},{ngram_range[1]}"

    print(f"Running experiment with max_features={max_features}, ngram_range=({ngram_range_str}) and combined_fields={combine_fields}...")

    # Measure the time it takes to run the experiment
    start_time = time()
    
    result = run(
        [
            "python3", "base.py", 
            "--max_features", str(max_features),
            "--ngram_range", ngram_range_str,
            "--combine_fields", combine_fields
        ], 
        capture_output=True,
        text=True
    )
    
    # Parse output to find accuracies
    output = result.stdout.splitlines()
    nb_accuracy = None
    svm_accuracy = None
    
    for line in output:
        if "Naive Bayes Accuracy" in line:
            nb_accuracy = float(line.split(":")[1].strip().replace("%", ""))
        elif "SVM Accuracy" in line:
            svm_accuracy = float(line.split(":")[1].strip().replace("%", ""))

    # Calculate the duration of the experiment
    duration = round(time() - start_time, 2)

    # Print how long in seconds it took to run the experiment
    print(f"Finished. Took {duration} seconds.")

    return nb_accuracy, svm_accuracy

# Run experiments
for combine_fields in combine_fields_values:
    for max_features in max_features_values:
        for ngram_range in ngram_ranges:
            # Label for the current experiment
            label = f"max_features={max_features}, ngram_range={ngram_range}, combine_fields={combine_fields}"
            experiment_labels.append(label)

            # Run the experiment and get accuracies
            nb_accuracy, svm_accuracy = run_experiment(max_features, ngram_range, combine_fields)
            
            # Store accuracies
            nb_accuracies.append(nb_accuracy)
            svm_accuracies.append(svm_accuracy)

print("All experiments completed. Plotting results...")

# Plot results
plt.figure(figsize=(12, 6))

# Adjust the labels to include newlines for better readability
formatted_labels = [label.replace(', ', '\n') for label in experiment_labels]
x = range(len(formatted_labels))

# Plot Naive Bayes and SVM accuracies with better styling
plt.plot(x, nb_accuracies, label="Naive Bayes Accuracy", marker='o', linestyle='-', color='blue', markersize=8)
plt.plot(x, svm_accuracies, label="SVM Accuracy", marker='x', linestyle='--', color='orange', markersize=8)

# Customize plot appearance
plt.xticks(x, formatted_labels, rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Experiments", fontsize=12, labelpad=15)
plt.ylabel("Accuracy (%)", fontsize=12, labelpad=15)
plt.title("Comparison of Naive Bayes and SVM Accuracies with Different Parameters", fontsize=14, pad=20)

# Add gridlines for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add legend with a better location and size
plt.legend(loc='best', fontsize=12)

# Add tight layout for better spacing and save the plot
plt.tight_layout()
plt.savefig('plots/experiment_results.png')  # Save the plot
