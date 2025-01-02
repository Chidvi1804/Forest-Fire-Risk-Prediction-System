import pandas as pd

# Create a sample dataset with X and Y values
data = {
    'X': [1, 2, 3, 4, 5, 6, 7, 8, 9],  # Sample X coordinates
    'Y': [2, 3, 4, 5, 6, 7, 8, 9, 9]   # Sample Y coordinates
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame to ensure it's correct
print(df)
