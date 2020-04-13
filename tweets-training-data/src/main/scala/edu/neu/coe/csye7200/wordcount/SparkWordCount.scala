package edu.neu.coe.csye7200.wordcount

import edu.neu.coe.csye7200.readcsv.readCsv.{readTrainData, sparksession}
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.rdd.RDD

import scala.collection.mutable

object SparkWordCount extends App {
  // read train data
  val train_data: DataFrame = readTrainData()

  // TF
  val hashingTF: HashingTF = new HashingTF()
    .setInputCol("filtered_words").setOutputCol("rawFeatures").setNumFeatures(200)
  val featurizedData: DataFrame = hashingTF.transform(train_data)
  featurizedData.show(false)
  // alternatively, CountVectorizer can also be used to get term frequency vectors

  // IDF
  val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")
  val idfModel: IDFModel = idf.fit(featurizedData)
  val rescaledData: DataFrame = idfModel.transform(featurizedData)

  // word count
  // filter real tweets and count frequencies
  val real_train_data: Dataset[Row] = rescaledData.filter("target == 1")
  var real_words_data: Seq[String] = Seq()
  real_train_data.foreach {
    row => {
      val filtered_words = row.toSeq(7)
      filtered_words match {
        case w: mutable.WrappedArray[String] => real_words_data ++= w
        case _ =>
      }
    }
  }

  val rdd_real_words: RDD[String] = sparksession.sparkContext.parallelize(real_words_data)
  val real_words_counts: RDD[(String, Int)] = rdd_real_words
    .map(word => (word, 1))
    .reduceByKey(_ + _).sortBy(_._2, false)
  real_words_counts.take(50).foreach(println)

  // filter fake tweets and count frequencies
  val fake_train_data: Dataset[Row] = rescaledData.filter("target == 0")
  var fake_words_data: Seq[String] = Seq()
  fake_train_data.foreach {
    row => {
      val filtered_words = row.toSeq(7)
      filtered_words match {
        case w: mutable.WrappedArray[String] => fake_words_data ++= w
        case _ =>
      }
    }
  }

  val rdd_fake_words: RDD[String] = sparksession.sparkContext.parallelize(fake_words_data)
  val fake_words_counts: RDD[(String, Int)] = rdd_fake_words
    .map(word => (word, 1))
    .reduceByKey(_ + _).sortBy(_._2, ascending = false)
  fake_words_counts.take(50).foreach(println)

  sparksession.stop()

//  val spark = SparkSession.builder
//    .master("local[*]")
//    .appName("Spark Word Count")
//    .getOrCreate()
//
//  val lines = spark.sparkContext.parallelize(
//    Seq("Spark Intellij Idea Scala test one",
//      "Spark Intellij Idea Scala test two",
//      "Spark Intellij Idea Scala test three"))
//
//  val counts = lines
//    .flatMap(line => line.split(" "))
//    .map(word => (word, 1))
//    .reduceByKey(_ + _)
//
//  counts.foreach(println)
}