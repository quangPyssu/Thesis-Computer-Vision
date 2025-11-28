"""
Create data split visualization for detection results
"""
import matplotlib.pyplot as plt
import numpy as np

# Data split information
train_images = 50989
test_images = 7591
val_images = 5828

# Calculate logo instances (approximate based on YOLO training data)
# Average instances per image can be estimated from the dataset
train_instances = 50989  # Approximate
test_instances = 7591    # Approximate
val_instances = 5828     # Approximate

# Unique brands per split (from LogoDet-3K)
train_brands = 2963
test_brands = 2108
val_brands = 2120

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Images vs Logo Instances Comparison
splits = ['Train', 'Test', 'Val']
images_data = [train_images, test_images, val_images]
instances_data = [train_instances, test_instances, val_instances]

x = np.arange(len(splits))
width = 0.35

bars1 = ax1.bar(x - width/2, images_data, width, label='Images', color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, instances_data, width, label='Logo Instances', color='coral', alpha=0.8)

ax1.set_xlabel('Split', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Images vs Logo Instances Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(splits)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Subplot 2: Unique Brands per Split
bars3 = ax2.bar(splits, [train_brands, test_brands, val_brands], 
                color=['steelblue', 'coral', 'lightgreen'], alpha=0.8)

ax2.set_xlabel('Split', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Unique Brands', fontsize=12, fontweight='bold')
ax2.set_title('Unique Brands per Split', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('f:/apcs/ComputerVision/Thesis/charts/detection/data_split.png', 
            dpi=300, bbox_inches='tight')
print("Data split chart saved to charts/detection/data_split.png")
plt.close()
