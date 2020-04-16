package edu.neu.coe.csye7200.unittests

import edu.neu.coe.csye7200.readcsv.readCsv.{readTestData, readTrainData}
import org.scalatest.{FlatSpec, Matchers}

class readCsvSpec extends FlatSpec with Matchers {
  behavior of "read csv files"

  it should "work for training data" in {
    val trainData = readTrainData()
    val size = trainData.count()
    println("trainData size: " + size)
    assert(size > 5000)
  }

  it should "work for testing data" in {
    val testData = readTestData()
    val size = testData._1.count()
    println("testData size: " + size)
    assert(size > 2000)
  }
}
