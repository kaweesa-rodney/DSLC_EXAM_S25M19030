import pandas as pd
import os
import numpy as np

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

#combined_df.to_csv('output/data/combined_data.csv', index=False)

print("Combined shape:", combined_df.shape)
    
#data cleaning

#celsius
def parse_temp(val):
    if pd.isna(val):
        return np.nan
    v = val.split(",")[0]
    if v in ["+9999", "9999"]:
        return np.nan
    return int(v) / 10.0


#m/s
def parse_wind(val):
    if pd.isna(val):
        return np.nan
    parts = val.split(",")
    speed = parts[3]
    if speed == "0000":
        return 0.0
    return int(speed) / 10.0


#metres
def parse_visibility(val):
    if pd.isna(val):
        return np.nan
    v = val.split(",")[0]
    return int(v)


#millimetres
def parse_precip(val):
    if pd.isna(val):
        return 0.0
    parts = val.split(",")
    if len(parts) < 2:
        return 0.0
    return int(parts[1]) / 10.0




# apply functions
combined_df["TMP_C"] = combined_df["TMP"].apply(parse_temp)
combined_df["DEW_C"] = combined_df["DEW"].apply(parse_temp)
combined_df["WIND_MS"] = combined_df["WND"].apply(parse_wind)
combined_df["VIS_M"] = combined_df["VIS"].apply(parse_visibility)
combined_df["PRECIP_MM"] = combined_df["AA1"].apply(parse_precip)

# Drop original encoded columns
combined_df = combined_df.drop(columns=["TMP", "DEW", "WND", "VIS", "AA1", "SLP"])

# Parse datetime
combined_df["DATE"] = pd.to_datetime(combined_df["DATE"])

# Save cleaned data
combined_df.to_csv("output/data/cleaned_weather.csv", index=False)

print("Cleaning complete.")
