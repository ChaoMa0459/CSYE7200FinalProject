package edu.neu.coe.csye7200.model

import edu.neu.coe.csye7200.readcsv.readCsv.{readTestData, readTrainData, sparksession}
import org.apache.spark.ml.feature.{HashingTF, IDF, IndexToString, RegexTokenizer, StopWordsRemover, StringIndexer, Tokenizer, VectorIndexer}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{DataType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

import scala.collection.mutable



  object Classification_models extends App {
    val result_train = readTrainData()
    val rescaledData = result_train._1
    val nums_of_features = readTrainData()._2
    val df = rescaledData.select("target","features")
    // Random Forest Classifier
    def RFC(): (DataFrame, Double) = {
    val labelIndexer = new StringIndexer().setInputCol("target").setOutputCol("indexedLabel").fit(df)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(nums_of_features).fit(df)
    val splits = df.randomSplit(Array(0.6, 0.4))
    val (train_data, test_data) = (splits(0), splits(1))
    print("There are "+train_data.count()+" rows in train data set\n")
    print("There are "+test_data.count()+" rows in test data set\n")
    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10).setMaxDepth(30).setSeed(5)
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    val model = pipeline.fit(train_data)
    val predictions = model.transform(test_data)
    predictions.select("predictedLabel", "target", "features").show(5)
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")
    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
      (predictions, accuracy)
    }
    // Naive Bayesian Classifier
    def NB(): (DataFrame, Double) = {
      def castColumnTo( df:DataFrame, cn:String, tpe:DataType ) : DataFrame = {
        df.withColumn( cn, df(cn).cast(tpe) )}
      val df_bayes = castColumnTo( df, "target", IntegerType )
      val labelIndexer = new StringIndexer().setInputCol("target").setOutputCol("indexedLabel").fit(df_bayes)
      val splits = df_bayes.randomSplit(Array(0.7, 0.3))
      val (train_data, test_data) = (splits(0), splits(1))
      val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
      val nb = new NaiveBayes().setLabelCol("target")
      val pipeline = new Pipeline().setStages(Array(labelIndexer, nb, labelConverter))
      val model = pipeline.fit(train_data)
      val predictions = model.transform(test_data)
      predictions.show()
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("target")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val accuracy = evaluator.evaluate(predictions)
      println(s"Test set accuracy = $accuracy")
      (predictions, accuracy)
    }
    NB()
    sparksession.stop()
  }