import pandas as pd

# Step 1: Load the full dataset
df = pd.read_csv("creditcard.csv")

# Step 2: Take a random sample (adjust number as needed)
sampled_df = df.sample(n=10000, random_state=42)  # Change 10000 as needed

# Step 3: Save to a new smaller CSV file
sampled_df.to_csv("creditcard_sample.csv", index=False)

print("Sampled CSV created successfully: creditcard_sample.csv")
