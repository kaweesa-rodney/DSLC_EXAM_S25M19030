from pyspark.sql import SparkSession
from pyspark.sql.functions import col, percentile_approx

spark = SparkSession.builder \
    .appName("HeatThresholdTuning") \
    .getOrCreate()

df = spark.read.csv(
    "output/data/cleaned_weather.csv",
    header=True,
    inferSchema=True
)

# Compute multiple candidate thresholds
thresholds = df.selectExpr(
    "percentile_approx(TMP_C, 0.90) as P90",
    "percentile_approx(TMP_C, 0.95) as P95",
    "percentile_approx(TMP_C, 0.97) as P97"
).collect()[0]


#taking P90 - have more positives and good for modelling
for p, val in zip(["P90", "P95", "P97"], thresholds):
    count = df.filter(col("TMP_C") > val).count()
    print(f"{p} threshold = {val:.2f} °C | Events = {count}")