package edu.neu.coe.csye7200.wordcount

// import edu.neu.coe.csye7200.wordcount.SparkWordCount.get_word_frequencies
import org.apache.spark.rdd.RDD
import org.scalatest.{FlatSpec, Matchers}

import scala.io.Source

class SparkWordCountSpec extends FlatSpec with Matchers {
  behavior of "word frequencies"

//  it should "work for real disaster tweets" in {
//    val real_words_counts = get_word_frequencies()._1
//    val size:Long = real_words_counts match {
//      case null => 0
//      case w: RDD[(String, Int)] => w.count()
//    }
//    assert(size > 0)
//  }
}
