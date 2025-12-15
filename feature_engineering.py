from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("HourlyFeatures").master("local[2]")\
    .config("spark.executor.cores", "1") \
    .config("spark.executor.memory", "2g").getOrCreate()

df = spark.read.csv("output/data/cleaned_weather.csv", header=True, inferSchema=True)

#derived column
df = df.withColumn("temp_dew_spread", col("tmp_c") - col("dew_c"))

df = df.withColumn("hour", hour("DATE"))
df = df.withColumn("day_of_year", dayofyear("DATE"))


# Rolling windows (hourly)
w3 = Window.orderBy("DATE").rowsBetween(-3, 0)
w6 = Window.orderBy("DATE").rowsBetween(-6, 0)
w12 = Window.orderBy("DATE").rowsBetween(-12, 0)
w24 = Window.orderBy("DATE").rowsBetween(-24, 0)
w72 = Window.orderBy("DATE").rowsBetween(-72, 0)

df = df.withColumn("rain_3h", sum("precip_mm").over(w3))
df = df.withColumn("rain_6h", sum("precip_mm").over(w6))
df = df.withColumn("rain_12h", sum("precip_mm").over(w12))
df = df.withColumn("rain_24h", sum("precip_mm").over(w24))
df = df.withColumn("rain_72h", sum("precip_mm").over(w72))


# Hourly rainfall anomaly
clim = df.groupBy("hour").agg(
    avg("precip_mm").alias("clim_mean"),
    stddev("precip_mm").alias("clim_std")
)

df = df.join(clim, on="hour", how="left")

df = df.withColumn(
    "rain_anomaly",
    try_divide(col("precip_mm") - col("clim_mean"), col("clim_std"))
)

df.show()

df.write.mode("overwrite").parquet("output/hourly_features")
spark.stop()