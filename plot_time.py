import json
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

# Load data from JSON file
results_file = 'results/safe_suffix_200.json'
with open(results_file, 'r') as file:
    data = json.load(file)

# Extract data
max_erase_values = []
time_per_prompt_values = []

for key, value in data.items():
    max_erase = int(key.split(': ')[1].replace('}', ''))
    max_erase_values.append(max_erase)
    time_per_prompt_values.append(value['time_per_prompt'])

# Plotting
sns.set_style("darkgrid", {"grid.color": ".85"})
plt.figure(figsize=(8, 6))
plt.plot(max_erase_values, time_per_prompt_values, linewidth=2)
plt.xlabel('Max Erased Tokens', fontsize=14, labelpad=10)
plt.ylabel('Time per Prompt (sec)', fontsize=14, labelpad=10)
plt.xlim(-0.1, 30.1)
plt.xticks(range(0, 31, 10))
plt.ylim(-0.01, 1.81)
plt.yticks(np.arange(0, 1.81, 0.3))
plt.grid(True, linewidth=2)
plt.tick_params(axis='both', labelsize=14)
plt.tight_layout()

# Save the figure
plot_file = results_file.replace('.json', '_time.png')
plt.savefig(plot_file, dpi=300)
plt.close()
