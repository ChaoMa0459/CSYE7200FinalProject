package edu.neu.coe.csye7200.wordcount

//import edu.neu.coe.csye7200.wordcount.SparkWordCount.filterData
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.{FlatSpec, Matchers}

import scala.io.Source

class SparkWordCountSpec extends FlatSpec with Matchers {
  behavior of "word frequencies"

  val filteredData: (Dataset[Row], Dataset[Row]) = filterData()

  it should "work for real disaster tweets count" in {
    val real_words = filteredData._1
    assert(real_words.count() == 3081)
  }

  it should "work for fake disaster tweets count" in {
    val fake_words = filteredData._2
    assert(fake_words.count() == 4095)
  }

  it should "work for real disaster tweets target" in {
    val target: String = filteredData._1.collect()(0).getAs("target")
    assert(target == "1")
  }

  it should "work for fake disaster tweets target" in {
    val target: String = filteredData._2.collect()(0).getAs("target")
    assert(target == "0")
  }
}
