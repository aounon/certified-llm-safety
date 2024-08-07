import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='Plot ROC curves from JSON file')
parser.add_argument('input_file', type=str, help='Path to the JSON input file')
parser.add_argument('--title', type=str, help='Title of the plot', default='ROC Curves')
args = parser.parse_args()

with open(args.input_file, 'r') as file:
    data = json.load(file)

save_file = args.input_file.split('.')[0] + '.png'

plt.figure(figsize=(7, 7))
sns.set_style("darkgrid")

for method, values in data.items():
    plt.plot(values['fpr'], values['tpr'], label=method, linewidth=2, marker='o')

plt.xlabel('False Positive Rate (%)', fontsize=18, labelpad=10)
plt.ylabel('True Positive Rate (%)', fontsize=18, labelpad=10)
plt.title(args.title, fontsize=20)
plt.legend(fontsize=14)
plt.xticks([i for i in range(0, 101, 20)], fontsize=14)
plt.yticks([i for i in range(0, 101, 20)], fontsize=14)
plt.grid(True, linewidth=2)
plt.tight_layout()
plt.savefig(save_file)
