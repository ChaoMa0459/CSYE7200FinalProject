package edu.neu.coe.csye7200.model

import edu.neu.coe.csye7200.readcsv.readCsv.{readTrainData, sparksession}
import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._

import scala.collection.mutable



  object Random_Forest extends App {
    val rescaledData: DataFrame = readTrainData()
    rescaledData.show(false)

    sparksession.stop()
  }
