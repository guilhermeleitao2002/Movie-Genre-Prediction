#!/usr/bin/env python3

import matplotlib.pyplot as plt
from os import makedirs

# Switch to the 'Agg' backend to avoid issues in headless environments
plt.switch_backend('Agg')


def plot(size, labels, accuracies, x_label, y_label, title, filename, mark, c):
    # Plot results
    plt.figure(figsize=size)

    # Adjust the labels to include newlines for better readability
    x = range(len(labels))

    # Plot Naive Bayes and SVM accuracies with better styling
    plt.plot(x, accuracies, label="SVM Accuracy", marker=mark, linestyle='--', color=c, markersize=8)

    # Customize plot appearance
    plt.xticks(x, labels, fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(x_label, fontsize=12, labelpad=15)
    plt.ylabel(y_label, fontsize=12, labelpad=15)
    plt.title(title, fontsize=14, pad=20)

    # Add gridlines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Add legend with a better location and size
    plt.legend(loc='best', fontsize=12)

    # Add tight layout for better spacing and save the plot
    plt.tight_layout()
    makedirs('plots', exist_ok=True) # Create the directory if it doesn't exist
    plt.savefig(f'plots/{filename}') # Save the plot


def main():
    # Plot the SVM accuracy vs. threshold
    plot(size=(8, 4),
        labels=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 1.00],
        accuracies=[71.43, 70.43, 70.19, 70.06, 70.43, 69.57, 67.20],
        x_label='Threshold',
        y_label='Accuracy (%)',
        title='SVM Accuracy vs. Threshold',
        filename='threshold.png',
        mark='o',
        c='orange')

    # Plot the SVM accuracy vs. n-gram range
    plot(size=(8, 4),
        labels=[(1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)],
        accuracies=[71.18, 70.43, 70.68, 71.43, 70.43, 70.43, 70.56],
        x_label='N-gram Range',
        y_label='Accuracy (%)',
        title='SVM Accuracy vs. N-gram Range',
        filename='ngram.png',
        mark='o',
        c='blue')

    # Print a success message
    print('All plots have been generated successfully.')


if __name__ == '__main__':
    main()
