import pandas as pd
import os

DATA_DIR = "data"

columns = [
    "STATION",
    "DATE",
    "LATITUDE",
    "LONGITUDE",
    "TMP",
    "DEW",
    "SLP",
    "WND",
    "VIS",
    "AA1"
]

dfs = []  # collect yearly dataframes

for f in os.listdir(DATA_DIR):
    if f.endswith(".csv"):
        file_path = os.path.join(DATA_DIR, f)
        print(f"Reading {file_path}")

        df = pd.read_csv(
            file_path,
            usecols=columns,      # only load needed columns
            low_memory=False
        )

        dfs.append(df)

# Combine all years
combined_df = pd.concat(dfs, ignore_index=True)

combined_df.to_csv('combined_data.csv', index=False)

print("Combined shape:", combined_df.shape)
    

