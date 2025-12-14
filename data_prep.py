import pandas as pd

rain = pd.read_csv("data/rainfall.csv")[[
    "Area", "Year", "average_rain_fall_mm_per_year"
]]

yield_df = pd.read_csv("data/yield.csv")[[
    "Area", "Item", "Year", "Value"
]].rename(columns={"Value": "Yield"})

temp = pd.read_csv("data/temp.csv").rename(columns={
    "country": "Area",
    "year": "Year"
})

pest = pd.read_csv("data/pesticides.csv")[[
    "Area", "Year", "Value"
]]


#combining data sets
df = yield_df.merge(rain, on=["Area","Year"], how="inner")
df = df.merge(temp, on=["Area","Year"], how="inner")
df = df.merge(pest, on=["Area","Year"], how="inner")

df = df.dropna()

#size of data set
df = df[df['Area']=='Uganda']
print(df.shape)


print(df.head(10))