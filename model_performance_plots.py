import matplotlib.pyplot as plt
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, percentile_approx, lag
from pyspark.sql.window import Window
from pyspark.ml import PipelineModel
from sklearn.metrics import roc_curve, auc

# --------------------------------------------------
# SPARK SESSION
# --------------------------------------------------
spark = SparkSession.builder \
    .appName("ExtremeEventROCPlots") \
    .getOrCreate()

# --------------------------------------------------
# LOAD DATA (SAME AS TRAINING)
# --------------------------------------------------
df = spark.read.csv(
    "output/data/cleaned_weather.csv",
    header=True,
    inferSchema=True
)

# --------------------------------------------------
# FEATURE ENGINEERING (EXACT MATCH)
# --------------------------------------------------
df = df.withColumn("TEMP_DEW_SPREAD", col("TMP_C") - col("DEW_C"))

w = Window.orderBy("DATE")

df = df.withColumn("TMP_LAG_1H", lag("TMP_C", 1).over(w))
df = df.withColumn("TMP_LAG_3H", lag("TMP_C", 3).over(w))
df = df.withColumn("TMP_LAG_6H", lag("TMP_C", 6).over(w))

df = df.dropna(subset=[
    "TMP_LAG_1H",
    "TMP_LAG_3H",
    "TMP_LAG_6H"
])

# --------------------------------------------------
# LABELS (SAME THRESHOLDS)
# --------------------------------------------------
heat_p90 = df.select(
    percentile_approx("TMP_C", 0.90)
).collect()[0][0]

rain_p95 = df.select(
    percentile_approx("PRECIP_MM", 0.95)
).collect()[0][0]

df = df.withColumn(
    "HEAT_EVENT",
    (col("TMP_C") > heat_p90).cast("int")
)

df = df.withColumn(
    "FLOOD_EVENT",
    (col("PRECIP_MM") > rain_p95).cast("int")
)

# --------------------------------------------------
# TEST SET (MATCH TRAINING SPLIT STYLE)
# --------------------------------------------------
test = df.sample(fraction=0.2, seed=42)

# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------
models = {
    "Heat": PipelineModel.load("outputs/models/heat_model"),
    "Flood": PipelineModel.load("outputs/models/flood_model")
}

# --------------------------------------------------
# ROC PLOTTING
# --------------------------------------------------
plt.figure(figsize=(8, 6))

for name, model in models.items():

    preds = model.transform(test)

    # 🔑 CRITICAL FIX: remove null labels
    preds = preds.dropna(subset=[f"{name.upper()}_EVENT"])

    # Convert to Pandas ONLY for sklearn plotting
    pdf = preds.select(
        col(f"{name.upper()}_EVENT").alias("label"),
        col("probability")
    ).toPandas()

    y_true = pdf["label"].values
    y_score = pdf["probability"].apply(lambda x: x[1]).values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} model (AUC = {roc_auc:.2f})"
    )

# --------------------------------------------------
# FINAL PLOT FORMATTING
# --------------------------------------------------
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Extreme Heat and Flood Prediction")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()