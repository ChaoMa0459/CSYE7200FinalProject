package edu.neu.coe.csye7200.readcsv

import org.scalatest.{FlatSpec, Matchers}
import edu.neu.coe.csye7200.readcsv.readCsv.{readTrainData, readTestData}

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
    val size = testData.count()
    println("testData size: " + size)
    assert(size > 2000)
  }
}
