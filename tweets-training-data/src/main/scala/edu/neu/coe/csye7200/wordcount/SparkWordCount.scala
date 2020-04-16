package edu.neu.coe.csye7200.wordcount

import edu.neu.coe.csye7200.readcsv.readCsv.{readTrainData, sparksession, clean_Data}
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{DoubleType, IntegerType, StringType, StructField, StructType}
import vegas.{Bar, Nom, Quant, Vegas}
import vegas.sparkExt._
import vegas.spec.Spec.TypeEnums.{Nominal, Quantitative}

import scala.collection.mutable


object SparkWordCount extends App {


  // read train data
  val rescaledData= clean_Data(readTrainData())

  rescaledData.show()
  // word count
  // filter real tweets and count frequencies
  val real_train_data: Dataset[Row] = rescaledData.filter("target == 1")

  def filterData(): (Dataset[Row], Dataset[Row]) = {
    // read train data
    val rescaledData: (DataFrame, Int) = readTrainData()

    rescaledData._1.show()
    // word count
    // filter real tweets and count frequencies
    val real_train_data: Dataset[Row] = rescaledData._1.filter("target == 1")
    val fake_train_data: Dataset[Row] = rescaledData._1.filter("target == 0")
    (real_train_data, fake_train_data)
  }

  val filteredData = filterData()
  val real_train_data = filteredData._1
  val fake_train_data = filteredData._2


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

  println("real_train_data: " + real_train_data.count())

  val size = real_words_data.size

  val rdd_real_words: RDD[String] = sparksession.sparkContext.parallelize(real_words_data)

  val real_words_counts: RDD[(String, Int)] = rdd_real_words
    .map(word => (word, 1))
    .reduceByKey(_ + _).sortBy(_._2, ascending = false)

  println("real_words_counts " + real_words_counts.count())

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

  println("fake_words_counts " + fake_words_counts.count())

  fake_words_counts.take(50).foreach(println)

  val real_count = real_train_data.count()

  val fake_count = fake_train_data.count()

  // plot read tweets count and fake tweets count
  val real_fake_count_plot = Vegas("Target values", width = 300.0, height = 500.0).
    withData(Seq(
      Map("tweets" -> "Real", "count" -> real_count), Map("tweets" -> "fake", "count" -> fake_count)
    )).
    mark(Bar).
    encodeX("tweets", Nom).
    encodeY("count", Quantitative)

  real_fake_count_plot.show

  // plot keyword count in real tweets
  val seq_real_words = real_words_counts.
    map(x => Map("real_tweets_words" -> x._1, "count" -> x._2)).collect().take(20)

  val real_word_count_plot = Vegas("Real words", width = 300.0, height = 500.0).
    withData(seq_real_words).
    mark(Bar).
    encodeX("real_tweets_words", Nom).
    encodeY("count", Quantitative)

  real_word_count_plot.show

  // plot keyword count in fake tweets
  val seq_fake_words = fake_words_counts.
    map(x => Map("fake_tweets_words" -> x._1, "count" -> x._2)).collect().take(20)

  val fake_word_count_plot = Vegas("Fake words", width = 300.0, height = 500.0).
    withData(seq_fake_words).
    mark(Bar).
    encodeX("fake_tweets_words", Nom).
    encodeY("count", Quantitative)

  fake_word_count_plot.show

}