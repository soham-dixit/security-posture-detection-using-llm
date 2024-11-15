# Assume these values are set in your TrainingArguments

import pandas as pd

# Load the sample CSV data
data_path = './UNSW_NB15_training-set.csv'
data = pd.read_csv(data_path)
total_samples = len(data)
batch_size = 4 * 1 # Adjust if using multiple GPUs
num_epochs = 3

total_steps = (total_samples // batch_size) * num_epochs
print("Total steps:", total_steps)
