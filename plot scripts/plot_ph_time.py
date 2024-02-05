import json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('results_file', type=str, help='Path to the results file')
args = parser.parse_args()

# Read data from JSON file
results_file = args.results_file
with open(results_file, 'r') as file:
    data = json.load(file)

# Extract data for plotting
num_iters = []
percent_harmful = []
time_per_prompt = []

for key, value in data.items():
    # Extract the number of iterations from the key string
    num_iter = int(key.split("'num_iters': ")[1].split("}")[0])
    num_iters.append(num_iter)
    percent_harmful.append(value["percent_harmful"])
    time_per_prompt.append(value["time_per_prompt"])

# Set seaborn style
sns.set(style="darkgrid")

# Create figure and axis objects with a shared x-axis
fig, ax1 = plt.subplots(figsize=(9, 6))

# Plot percent_harmful
color = 'tab:red'
ax1.set_xlabel('# Iterations', fontsize=18, labelpad=10)
ax1.set_xticks(num_iters)
# Set x-axis limits
ax1.set_xlim([num_iters[0], num_iters[-1]])
ax1.set_ylabel('Percent Harmful', color=color, fontsize=18, labelpad=10)
ax1.set_yticks(range(0, 5, 1))
ax1.set_ylim([0, 4])
# ax1.set_ylabel('Percent Harmful', fontsize=18, labelpad=10)
ax1.plot(num_iters, percent_harmful, color=color, marker='o')
# ax1.plot(num_iters, percent_harmful)
ax1.tick_params(axis='y', labelcolor=color, labelsize=18)
# ax1.tick_params(axis='y', labelsize=18)
ax1.tick_params(axis='x', labelsize=18)

# Create a second y-axis to share the same x-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Time / Prompt', color=color, fontsize=18, labelpad=10)  # we already handled the x-label with ax1
ax2.set_yticks(np.arange(0, 0.051, 0.01))
ax2.plot(num_iters, time_per_prompt, color=color, marker='o')
# ax2.plot(num_iters, time_per_prompt)
ax2.tick_params(axis='y', labelcolor=color, labelsize=18)
# ax2.tick_params(axis='y', labelsize=18)

# Save the figure
fig.tight_layout()
save_path = results_file.split(".")[0] + ".png"
plt.savefig(save_path)
