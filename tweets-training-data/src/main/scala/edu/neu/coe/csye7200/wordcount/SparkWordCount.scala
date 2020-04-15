package edu.neu.coe.csye7200.wordcount

import edu.neu.coe.csye7200.readcsv.readCsv.{readTrainData, sparksession}
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.rdd.RDD
import vegas.{Bar, Nom, Quant, Vegas}
import vegas.sparkExt._
import vegas.spec.Spec.TypeEnums.{Nominal, Quantitative}

import scala.collection.mutable


object SparkWordCount extends App {

  // read train data
  val rescaledData:(DataFrame, Int)= readTrainData()

  rescaledData._1.show()
  // word count
  // filter real tweets and count frequencies
  val real_train_data: Dataset[Row] = rescaledData._1.filter("target == 1")

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

  println(real_train_data)

  val rdd_real_words: RDD[String] = sparksession.sparkContext.parallelize(real_words_data)
  val real_words_counts: RDD[(String, Int)] = rdd_real_words
    .map(word => (word, 1))
    .reduceByKey(_ + _).sortBy(_._2, ascending = false)
  println("real_words_counts " + real_words_counts.count())
  real_words_counts.take(50).foreach(println)

  // filter fake tweets and count frequencies
  val fake_train_data: Dataset[Row] = rescaledData._1.filter("target == 0")
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

  // (real_words_counts, fake_words_counts)

  val real_count = real_train_data.count()

  val fake_count = fake_train_data.count()

  val real_fake_count_plot = Vegas("Target values").
  withData(Seq(
    Map("name" -> "Real", "count" -> real_count), Map("name" -> "fake", "count" -> fake_count)
  )).
    mark(Bar).
    encodeX("name", Nom).
    encodeY("count", Quantitative)

  real_fake_count_plot.show

  val plot = Vegas("Country Pop").
    withDataFrame(real_train_data.limit(20)).
    encodeX("text", Nom).
    encodeY("target", Quant).
    mark(Bar)

  plot.show

}