package edu.neu.coe.csye7200.model

import edu.neu.coe.csye7200.readcsv.readCsv.{readTrainData, sparksession}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, NaiveBayes, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{DataType, DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.classification.LinearSVC
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import vegas.{Bar, Nom, Quant, Vegas}

import scala.collection.mutable




  object Classification_models extends App {
    val result_train = readTrainData()
    val rescaledData = result_train._1
    val nums_of_features = readTrainData()._2
    val df = rescaledData.select("target","features")
    // Random Forest Classifier
    def split_data(df:DataFrame): (DataFrame, DataFrame) = {
      val splits = df.randomSplit(Array(0.6, 0.4))
      val (train_data, test_data) = (splits(0), splits(1))
//      print("There are "+train_data.count()+" rows in train data set\n")
//      print("There are "+test_data.count()+" rows in test data set\n")
      (train_data,test_data)
    }
    def castColumnTo( df:DataFrame, cn:String, tpe:DataType ) : DataFrame = {
      df.withColumn( cn, df(cn).cast(tpe) )}

    def RFC(): (DataFrame, Double) = {
    val labelIndexer = new StringIndexer().setInputCol("target").setOutputCol("indexedLabel").fit(df)
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(nums_of_features).fit(df)
    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10).setMaxDepth(30).setSeed(5)
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
    val model = pipeline.fit(split_data(df)._1)
    val predictions = model.transform(split_data(df)._2)
    predictions.select("predictedLabel", "target", "features").show(5)
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test set accuracy of Random Forest Classifier= ${accuracy}")
    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
      (predictions, accuracy)
    }
    // Naive Bayesian Classifier
    def NB(): (DataFrame, Double) = {
      val df_bayes = castColumnTo( df, "target", IntegerType )
      val labelIndexer = new StringIndexer().setInputCol("target").setOutputCol("indexedLabel").fit(df_bayes)
      val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
      val nb = new NaiveBayes().setLabelCol("target").setSmoothing(10)
      val pipeline = new Pipeline().setStages(Array(labelIndexer, nb, labelConverter))
      val model = pipeline.fit(split_data(df_bayes)._1)
      val predictions = model.transform(split_data(df_bayes)._2)
      predictions.show()
      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("target")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")
      val accuracy = evaluator.evaluate(predictions)
      println(s"Test set accuracy of Naive Bayesian = $accuracy")
      (predictions, accuracy)
    }
    def SVC(): (DataFrame, Double) = {
      val df_svc = castColumnTo( df, "target", IntegerType )
      val lsvc = new LinearSVC().setLabelCol("target").setMaxIter(50).setRegParam(0.1).setStandardization(false).setFitIntercept(true)
      val lsvcModel = lsvc.fit(split_data(df_svc)._1)
      val predictions = lsvcModel.transform(split_data(df_svc)._2)
      val evaluator = new MulticlassClassificationEvaluator().setLabelCol("target").setPredictionCol("prediction").setMetricName("accuracy")
      val accuracy = evaluator.evaluate(predictions)
      println(s"Test set accuracy of Linear SVC= ${accuracy}")
      (predictions, accuracy)
    }
    println("Start training and testing Random Forest Classifier")
    val rfc_res = RFC()
    println("Start training and testing Naive Bayesian Classifier")
    val nb_res = NB()
    println("Start training and testing Linear Support Vector Classifier")
    val svc_res = SVC()

    val arraylist: Array[(String, Any)] = Array(("Random Forest Classifier",rfc_res._2),
                                                ("Naive Bayesian Classifier",nb_res._2),
                                                ("Linear Support Vector Classifier",svc_res._2));

    val schema = StructType(
      StructField("Model", StringType, false) ::
      StructField("Accuracy", DoubleType, false) :: Nil)
    val rdd = sparksession.sparkContext.parallelize (arraylist).map (x => Row(x._1, x._2.asInstanceOf[Number].doubleValue()))
    val sqlContext = new org.apache.spark.sql.SQLContext(sparksession.sparkContext)
    val df_res = sqlContext.createDataFrame(rdd, schema)

    df_res.show()

//    // plot model and accuracy
//    val seq_res: Seq[Map[String, Double]] = df_res.collect().
//      map(x => Map(
//          "Model" -> x.getAs("Model"),
//          "Accuracy" -> x.getAs("Accuracy"))
//      )
//    val res_plot = Vegas("Model accuracy", width = 300.0, height = 500.0).
//      withData(seq_res).
//      encodeX("Model", Nom).
//      encodeY("Accuracy", Quant).
//      mark(Bar)
//    res_plot.show

    sparksession.stop()
  }