import pandas as pd

# Load the original dataset
data_path = './UNSW_NB15_training-set.csv'
data = pd.read_csv(data_path)

# Drop duplicates, keeping only the first occurrence of each unique 'attack_cat'
unique_attack_data = data.drop_duplicates(subset=['attack_cat'], keep='first')

# Save the result to a new CSV file
output_path = './unique_attack_cats.csv'
unique_attack_data.to_csv(output_path, index=False)

print(f"New CSV file with unique attack categories saved to {output_path}")
