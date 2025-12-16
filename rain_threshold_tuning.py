from pyspark.sql import SparkSession
from pyspark.sql.functions import col, percentile_approx

spark = SparkSession.builder \
    .appName("RainThresholdTuning") \
    .getOrCreate()

df = spark.read.csv(
    "output/data/cleaned_weather.csv",
    header=True,
    inferSchema=True
)

thresholds = df.selectExpr(
    "percentile_approx(PRECIP_MM, 0.90) as P90",
    "percentile_approx(PRECIP_MM, 0.95) as P95",
    "percentile_approx(PRECIP_MM, 0.99) as P99"
).collect()[0]


#taking P95 - has a good balance in terms of precipitation
for p, val in zip(["P90", "P95", "P99"], thresholds):
    count = df.filter(col("PRECIP_MM") > val).count()
    print(f"{p} threshold = {val:.2f} mm | Events = {count}")