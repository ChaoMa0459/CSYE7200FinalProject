package edu.neu.coe.csye7200.model

import org.apache.spark
import org.apache.spark.sql.SparkSession
import vegas._
import vegas.sparkExt._
import vegas.render.WindowRenderer._

object VegasPlot extends App {
  val sparksession: SparkSession = org.apache.spark.sql.SparkSession.builder
    .master("local")
    .appName("Spark CSV Reader")
    .getOrCreate;

  val plot = Vegas("Country Pop").
    withData(
      Seq(
        Map("country" -> "USA", "population" -> 314),
        Map("country" -> "UK", "population" -> 64),
        Map("country" -> "DK", "population" -> 80)
      )
    ).
    encodeX("country", Nom).
    encodeY("population", Quant).
    mark(Bar)

  plot.show

  val df = sparksession.createDataFrame(Seq(("A", 12), ("B", 24), ("C", 16))).toDF("name", "count")

  Vegas("A simple bar chart with spark data frame.").
    withDataFrame(df).
    encodeX("name", Ordinal).
    encodeY("count", Quantitative).
    mark(Bar).
    show

}
