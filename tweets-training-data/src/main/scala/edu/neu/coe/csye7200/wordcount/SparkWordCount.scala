package edu.neu.coe.csye7200.wordcount

import edu.neu.coe.csye7200.readcsv.readCsv.{readTrainData, sparksession}
import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.rdd.RDD
import vegas.{Bar, Nom, Quant, Vegas}
import vegas.sparkExt._

import scala.collection.mutable


object SparkWordCount extends App {

  // read train data
  val rescaledData:(DataFrame, Int)= readTrainData()

//  // TF
//  val hashingTF: HashingTF = new HashingTF()
//    .setInputCol("filtered_words").setOutputCol("rawFeatures").setNumFeatures(200)
//  val featuredData: DataFrame = hashingTF.transform(train_data)
//  featuredData.show(false)
//  // alternatively, CountVectorizer can also be used to get term frequency vectors
//
//  // IDF
//  val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")
//  val idfModel: IDFModel = idf.fit(featuredData)
//  val rescaledData: DataFrame = idfModel.transform(featuredData)

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

  println("end function")

  // (real_words_counts, fake_words_counts)

  val plot = Vegas("Country Pop").
    withDataFrame(real_train_data.limit(20)).
    encodeX("text", Nom).
    encodeY("target", Quant).
    mark(Bar)

  plot.show

}