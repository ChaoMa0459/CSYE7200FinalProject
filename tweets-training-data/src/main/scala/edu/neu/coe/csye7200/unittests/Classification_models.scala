package edu.neu.coe.csye7200.unittests

import edu.neu.coe.csye7200.readcsv.readCsv.{readTrainData, clean_Data, tf_idf, sparksession}
import sparksession.implicits._
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, RegexTokenizer, StopWordsRemover, Tokenizer}
import edu.neu.coe.csye7200.swing.GUI
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LinearSVC, NaiveBayes, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{DataType, DoubleType, IntegerType, StringType, StructField, StructType}
import org.apache.spark.ml.feature.{IndexToString, RegexTokenizer, StopWordsRemover, StringIndexer, Tokenizer, VectorIndexer}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.expressions.UserDefinedFunction
import vegas.{Bar, Nom, Quant, Vegas}

import scala.collection.mutable




  object Classification_models extends App {


    def build_model(text: String, model: String, num_of_tree: String, max_depth: String, seed: String,
                    smoothing: String, max_iter: String, regularization: String, standardize: String, fit: String): (Double, String) = {

      val data = readTrainData()
      val cleaned_data = clean_Data(data)
      val user_df = Seq(("user", "null", "null", text, "1")).toDF("id", "keyword", "location", "text", "target")
      val cleaned_user_data = clean_Data(user_df)
      val tf_idf_res = tf_idf(cleaned_data.union(cleaned_user_data))
      val featured_data_user = tf_idf_res._1
      val num_features = tf_idf_res._2
      val featured_user =featured_data_user.filter(featured_data_user("id") === "user")
      val featured_df = featured_data_user.filter(featured_data_user("id") !== "user")

      def split_data(df: DataFrame): (DataFrame, DataFrame) = {
        val splits = df.randomSplit(Array(0.6, 0.4))
        val (train_data, test_data) = (splits(0), splits(1))
        //      print("There are "+train_data.count()+" rows in train data set\n")
        //      print("There are "+test_data.count()+" rows in test data set\n")
        (train_data, test_data)
      }
      // Random Forest Classifier

      def castColumnTo(df: DataFrame, cn: String, tpe: DataType): DataFrame = {
        df.withColumn(cn, df(cn).cast(tpe))
      }

      def RFC(df:DataFrame, user:DataFrame, nums_feature: Int):(DataFrame, Double, String) = {
        var string_res = ""
        val labelIndexer = new StringIndexer().setInputCol("target").setOutputCol("indexedLabel").fit(df)
        val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(nums_feature).fit(df)
        val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(num_of_tree.toInt).setMaxDepth(max_depth.toInt).setSeed(seed.toInt)
        val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
        val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
        val (train_df, test_df) = split_data(df)
        val model = pipeline.fit(train_df)
        val predictions = model.transform(test_df.union(user))
        val predic_user = predictions.filter((predictions("id") === "user")).select(col("predictedLabel")).first.getString(0)
        if (predic_user == "0") {string_res = "This tweet is not reporting a disaster."}
        else if(predic_user == "1") {string_res = "This tweet is reporting a disaster."}
        else {string_res = "I can not figure it out."}
        val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println(s"Test set accuracy of Random Forest Classifier= ${accuracy}")
        val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
        // println(s"Learned classification forest model:\n ${rfModel.toDebugString}")

        (predictions, accuracy, string_res)
      }

      // Naive Bayesian Classifier
      def NB(df:DataFrame, user:DataFrame): (DataFrame, Double, String) = {
        var string_res = ""
        val df_bayes = castColumnTo(df, "target", IntegerType)
        val df_user = castColumnTo(user, "target", IntegerType)
        val nb = new NaiveBayes().setLabelCol("target").setSmoothing(smoothing.toDouble)
        val pipeline = new Pipeline().setStages(Array(nb))
        val (train_df, test_df) = split_data(df_bayes)
        val model = pipeline.fit(train_df)
        val predictions = model.transform(test_df.union(df_user))
        val predic_user = predictions.filter((predictions("id") === "user")).select(col("prediction")).first.getDouble(0).toInt.toString
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("target")
          .setPredictionCol("prediction")
          .setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println(s"Test set accuracy of Naive Bayesian = $accuracy")
        if (predic_user == "0") {string_res = "This tweet is not reporting a disaster."}
        else if(predic_user == "1") {string_res = "This tweet is reporting a disaster."}
        else {string_res = "I can not figure it out."}
        (predictions, accuracy, string_res)
      }

      def SVC(df:DataFrame, user:DataFrame): (DataFrame, Double, String) = {
        var string_res = ""
        val df_svc = castColumnTo(df, "target", IntegerType)
        val df_user = castColumnTo(user, "target", IntegerType)
        val lsvc = new LinearSVC().setLabelCol("target").setMaxIter(max_iter.toInt)
          .setRegParam(regularization.toDouble).setStandardization(standardize.toBoolean).setFitIntercept(fit.toBoolean)
        val (train_df, test_df) = split_data(df_svc)
        val lsvcModel = lsvc.fit(train_df)
        val predictions = lsvcModel.transform(test_df.union(df_user))
        val predic_user = predictions.filter((predictions("id") === "user")).select(col("prediction")).first.getDouble(0).toInt.toString
        val evaluator = new MulticlassClassificationEvaluator().setLabelCol("target").setPredictionCol("prediction").setMetricName("accuracy")
        val accuracy = evaluator.evaluate(predictions)
        println(s"Test set accuracy of Linear SVC= ${accuracy}")
        if (predic_user == "0") {string_res = "This tweet is not reporting a disaster."}
        else if(predic_user == "1") {string_res = "This tweet is reporting a disaster."}
        else {string_res = "I can not figure it out."}
        (predictions, accuracy, string_res)
      }

      if(model == "Random Forest Classifier"){
        println("Start training and testing Random Forest Classifier")
        val rfc_res = RFC(featured_df, featured_user, num_features.toInt)
        rfc_res._1.show()
        println(rfc_res._2)
        println(rfc_res._3)
        return (rfc_res._2, rfc_res._3)
      }
      if(model == "Naive Bayesian Classifier"){
        println("Start training and testing Naive Bayesian Classifier")
        val nb_res = NB(featured_df, featured_user)
        nb_res._1.show()
        println(nb_res._2)
        println(nb_res._3)
        return (nb_res._2, nb_res._3)
      }
      if(model == "SVM"){
        println("Start training and testing Linear Support Vector Classifier")
        val svc_res = SVC(featured_df, featured_user)
        svc_res._1.show()
        println(svc_res._2)
        println(svc_res._3)
        return (svc_res._2, svc_res._3)
      }
      else
        {
          return (0, "Please select a model!")
        }
  //    val arraylist: Array[(String, Any)] = Array(("Random Forest Classifier", rfc_res._2),
  //      ("Naive Bayesian Classifier", nb_res._2),
  //      ("Linear Support Vector Classifier", svc_res._2));
  //
  //    val schema = StructType(
  //      StructField("Model", StringType, false) ::
  //        StructField("Accuracy", DoubleType, false) :: Nil)
  //    val rdd = sparksession.sparkContext.parallelize(arraylist).map(x => Row(x._1, x._2.asInstanceOf[Number].doubleValue()))
  //    val sqlContext = new org.apache.spark.sql.SQLContext(sparksession.sparkContext)
  //    val df_res = sqlContext.createDataFrame(rdd, schema)
  //
  //    df_res.show()

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

//      sparksession.stop()
    }
  }