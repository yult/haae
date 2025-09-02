# -*- coding: utf-8 -*-
## setup
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config_loader import get_data_dir

measurements_filepath = os.path.join(get_data_dir(), 'output')
measurements_filename = os.path.join(get_data_dir(), 'sample/measurements.csv')
## load measurements data
data_measurements = pd.read_csv(measurements_filename, encoding='utf-8', dtype={
    'Model': str,
    'Train AUC': np.float32,
    'Test Accuracy': np.float32,
    'Test Precision': np.float32,
    'Test Recall': np.float32,
    'Test F1 Score': np.float32,
    'Test AUC': np.float32,
    'ΔAUC (Train - Test)': np.float32
})
data_measurements.head()

## plot  measurements
# Melt the dataframe to long format for seaborn
melted_data = data_measurements.melt(
    id_vars='Model',
    value_vars=['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1 Score', 'Test AUC'],
    var_name='Metric',
    value_name='Value'
)

melted_data_auc = data_measurements.melt(
    id_vars='Model',
    value_vars=['Train AUC', 'Test AUC', 'ΔAUC (Train - Test)'],
    var_name='Metric',
    value_name='Value'
)

# Create combined figure
plt.figure(figsize=(12, 12))
plt.rcParams.update({'font.size': 12})
sns.set_style("whitegrid")
palette = sns.color_palette("YlGnBu")

# Create subplots grid
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)

# 1. Performance Metrics Comparison
sns.lineplot(
    data=melted_data,
    x='Metric',
    y='Value',
    hue='Model',
    palette=palette,
    marker='o',
    markersize=8,
    linewidth=2.5,
    markeredgecolor='black',
    markeredgewidth=1,
    ax=ax1
)

# Add value labels
j_ax1 = -1
for line in ax1.lines:
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    j_ax1 = j_ax1 * (-1)
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        x_offset = 0.035 * j_ax1
        y_offset = 0.01
        ax1.text(
            x + x_offset,
            y + y_offset,
            f'{y:.2f}',
            ha='center',
            va='bottom',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2')
        )

ax1.set_title("(a) Performance Metrics Comparison", fontsize=14, y=-0.25, pad=25)
ax1.set_xlabel('Metrics', fontsize=12, labelpad=10)
ax1.set_ylabel('Score Value', fontsize=12, labelpad=10)
ax1.set_xticklabels(melted_data['Metric'].unique(), ha='right')
ax1.yaxis.grid(True, linestyle='--', alpha=0.6)

# 2. AUC Metrics Comparison
sns.lineplot(
    data=melted_data_auc,
    x='Metric',
    y='Value',
    hue='Model',
    palette=palette,
    marker='o',
    markersize=8,
    linewidth=2.5,
    markeredgecolor='black',
    markeredgewidth=1,
    ax=ax2
)

# Add value labels
j_ax2 = -1
for line in ax2.lines:
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    j_ax2 = j_ax2 * (-1)
    for i, (x, y) in enumerate(zip(x_data, y_data)):
        x_offset = 0.035 * j_ax2
        y_offset = 0.01
        ax2.text(
            x + x_offset,
            y + y_offset,
            f'{y:.2f}',
            ha='center',
            va='bottom',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2')
        )

ax2.set_title("(b) AUC Metrics Comparison", fontsize=14, y=-0.25, pad=25)
ax2.set_xlabel('Metrics', fontsize=12, labelpad=10)
ax2.set_ylabel('AUC Value', fontsize=12, labelpad=10)
ax2.set_xticklabels(melted_data_auc['Metric'].unique(), ha='right')
ax2.yaxis.grid(True, linestyle='--', alpha=0.6)

# Unified legend
legend_style = {
    'title': 'Model',
    'frameon': True,
    'edgecolor': 'black',
    'facecolor': 'white',
    'fontsize': 12,
    'title_fontsize': 12,
    'handletextpad': 1.0,
    'markerscale': 1.0
}

ax1.legend(
    loc='lower right',
    bbox_to_anchor=(1, 0),
    **legend_style
)

ax2.legend(
    loc='lower left',
    bbox_to_anchor=(0, 0),
    **legend_style
)

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)

    legend = ax.get_legend()
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_alpha(1.0)

# Adjust layout
plt.tight_layout()
plt.subplots_adjust(bottom=0.2, wspace=0.3)

# Save combined plot
plt.savefig(os.path.join(measurements_filepath, 'combined_metrics_comparison.png'),
            bbox_inches='tight', dpi=300)
plt.close()
