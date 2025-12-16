from pyspark.sql import SparkSession
from pyspark.sql.functions import col, percentile_approx, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("DualHazardTraining") \
    .getOrCreate()


df = spark.read.csv(
    "output/data/cleaned_weather.csv",
    header=True,
    inferSchema=True
)

# --------------------------------------------------
# DERIVED FEATURES
# --------------------------------------------------
df = df.withColumn("TEMP_DEW_SPREAD", col("TMP_C") - col("DEW_C"))

# --------------------------------------------------
# LABELS / OUTCOMES (TROPICAL THRESHOLDS)
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
# FEATURES (NO LEAKAGE)
# --------------------------------------------------
heat_features = [
    "DEW_C",
    "TEMP_DEW_SPREAD",
    "WIND_MS",
    "VIS_M"
]

flood_features = [
    "VIS_M",
    "WIND_MS",
    "TEMP_DEW_SPREAD",
    "DEW_C"
]

# --------------------------------------------------
# HANDLE NULLS
# --------------------------------------------------
required_cols = list(set(
    heat_features + flood_features + ["HEAT_EVENT", "FLOOD_EVENT"]
))

df = df.dropna(subset=required_cols)
print("Rows after null handling:", df.count())

# --------------------------------------------------
# CLASS WEIGHTS (IMBALANCE HANDLING)
# --------------------------------------------------
heat_pos = df.filter(col("HEAT_EVENT") == 1).count()
heat_neg = df.filter(col("HEAT_EVENT") == 0).count()

flood_pos = df.filter(col("FLOOD_EVENT") == 1).count()
flood_neg = df.filter(col("FLOOD_EVENT") == 0).count()

df = df.withColumn(
    "heat_weight",
    when(col("HEAT_EVENT") == 1, heat_neg / heat_pos).otherwise(1.0)
)

df = df.withColumn(
    "flood_weight",
    when(col("FLOOD_EVENT") == 1, flood_neg / flood_pos).otherwise(1.0)
)

# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)

# --------------------------------------------------
# PIPELINE TRAINING FUNCTION (NO SCALING)
# --------------------------------------------------
def train_model(feature_cols, label_col, model):
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="assembled",
        handleInvalid="skip"
    )

    model.setFeaturesCol("assembled")
    model.setLabelCol(label_col)

    pipeline = Pipeline(stages=[assembler, model])
    return pipeline.fit(train)

# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------
heat_lr = train_model(
    heat_features,
    "HEAT_EVENT",
    LogisticRegression(
        maxIter=50,
        regParam=0.3,
        elasticNetParam=0.8,
        weightCol="heat_weight"
    )
)

heat_rf = train_model(
    heat_features,
    "HEAT_EVENT",
    RandomForestClassifier(
        numTrees=30,
        maxDepth=3,
        minInstancesPerNode=100,
        featureSubsetStrategy="sqrt",
        seed=42
    )
)

flood_lr = train_model(
    flood_features,
    "FLOOD_EVENT",
    LogisticRegression(
        maxIter=50,
        regParam=0.3,
        elasticNetParam=0.8,
        weightCol="flood_weight"
    )
)

flood_rf = train_model(
    flood_features,
    "FLOOD_EVENT",
    RandomForestClassifier(
        numTrees=30,
        maxDepth=3,
        minInstancesPerNode=100,
        featureSubsetStrategy='sqrt',
        seed=42
    )
)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------
def make_evaluator(label_col):
    return BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

def eval_model(model, label_col):
    evaluator = make_evaluator(label_col)
    return evaluator.evaluate(model.transform(test))

metrics = {
    "Heat_LR": eval_model(heat_lr, "HEAT_EVENT"),
    "Heat_RF": eval_model(heat_rf, "HEAT_EVENT"),
    "Flood_LR": eval_model(flood_lr, "FLOOD_EVENT"),
    "Flood_RF": eval_model(flood_rf, "FLOOD_EVENT")
}

for k, v in metrics.items():
    print(f"{k} AUC: {v:.3f}")

"""
heat_best = heat_rf if metrics["Heat_RF"] > metrics["Heat_LR"] else heat_lr
flood_best = flood_rf if metrics["Flood_RF"] > metrics["Flood_LR"] else flood_lr

heat_best.write().overwrite().save("outputs/models/heat_model")
flood_best.write().overwrite().save("outputs/models/flood_model")
"""
