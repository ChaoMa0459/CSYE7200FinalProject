package edu.neu.coe.csye7200.model

import edu.neu.coe.csye7200.readcsv.readCsv.{readTrainData, readTestData, sparksession}
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer, VectorIndexer, StringIndexer, IndexToString}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._

import scala.collection.mutable



  object Random_Forest extends App {
    val result_train = readTrainData()
    println(result_train)
    val rescaledData = result_train._1
    val nums_of_features = readTrainData()._2
    rescaledData.show(false)
    val df = rescaledData.select("target","features")
    // df.show()
    val splits = df.randomSplit(Array(0.7, 0.3))
    val (train_data, test_data) = (splits(0), splits(1))
    train_data.show()
    test_data.show()
    print("There are "+train_data.count()+" rows in train data set\n")
    print("There are "+test_data.count()+" rows in test data set\n")

    val labelIndexer = new StringIndexer()
      .setInputCol("target")
      .setOutputCol("indexedLabel")
      .fit(df)
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(nums_of_features)
      .fit(df)
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    val model = pipeline.fit(train_data)
    val predictions = model.transform(test_data)
    predictions.select("predictedLabel", "target", "features").show(5)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
    sparksession.stop()

  }
