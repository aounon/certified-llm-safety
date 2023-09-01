import json
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('results_file', type=str,
                    help='Path to the results file')
args = parser.parse_args()

# Read data from JSON file
results_file = args.results_file
with open(results_file, 'r') as file:
    data = json.load(file)

# Extract data
max_erase_values = []
percent_safe_values = []

for key, value in data.items():
    max_erase = int(key.split(': ')[1].replace('}', ''))
    max_erase_values.append(max_erase)
    percent_safe_values.append(value['percent_safe'])

# Plotting
sns.set_style("darkgrid", {"grid.color": ".85"})
plt.figure(figsize=(8, 6))
plt.plot(max_erase_values, percent_safe_values, '-', color='tab:green', label='Safe Prompts (%, empirical)', linewidth=2)
plt.axhline(y=93.6, color='tab:blue', linewidth=2, linestyle='--', label='Certified Harmful Prompts (93.6%)')
plt.xlabel('Max Erased Tokens (= Certified Length)', fontsize=14, labelpad=10)
plt.ylabel('Accuracy (%)', fontsize=14)
plt.xlim(-0.05, 12.05)
plt.xticks(range(0, 13, 4))
# plt.xlim(-0.1, 30.1)
# plt.xticks(range(0, 31, 10))
plt.ylim(-0.5, 100.5)
plt.grid(True, linewidth=2)
plt.legend(loc="lower right",fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

# Save the figure
plot_file = results_file.replace('.json', '_acc.png')
plt.savefig(plot_file, dpi=300)
plt.close()
