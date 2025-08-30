import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Choose a style for the plot
plt.style.use('default')

# Load loss data
loss_data = pd.read_csv('training_losses.csv')
epochs = loss_data['epoch']
training_loss = loss_data['train_loss']
validation_loss = loss_data['val_loss']

# Create figure
plt.figure(figsize=(10, 7))

# Plot training and validation loss
plt.plot(epochs, training_loss, label='Training Loss',
         color='#ffda06', linestyle='-', linewidth=2.2, alpha=1)

plt.plot(epochs, validation_loss, label='Validation Loss',
         color='red', linestyle='--', linewidth=2, alpha=1)

# Labels and title
plt.xlabel('Epochs', fontsize=14, fontweight='bold', color='#333333')
plt.ylabel('Loss', fontsize=14, fontweight='bold', color='#333333')
plt.title('Training and Validation Loss', fontsize=17, fontweight='bold', color='#222222')

plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
plt.legend(loc='upper right', fontsize=13, frameon=False)

# Adjust layout for a clean look
plt.tight_layout()
plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# Load confusion matrix data
conf_data = pd.read_csv('confusion_matrix.csv', index_col=0)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(10, 7))  # Set figure size

# Plot the confusion matrix with annotations and class names
sns.heatmap(conf_data, annot=True, fmt='d', cmap='viridis')

# Add labels and title to the plot
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
# Show the plot
plt.show()
