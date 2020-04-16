package edu.neu.coe.csye7200.unittests

import edu.neu.coe.csye7200.unittests.Classification_models.build_model
import org.scalatest.{FlatSpec, Matchers}

class modelSpec extends FlatSpec with Matchers {

//  val result = build_model(text, model, num_of_tree, max_depth, seed, smoothing, max_iter, regularization, standardize, fit)

  behavior of "model accuracy"

  val text = "Wreckage 'Conclusively Confirmed' as From MH370: Malaysia PM http://t.co/MN130C4e2D via @ndtv"
  val num_of_tree = "30"
  val max_depth = "30"
  val seed = "5"
  val smoothing = "10"
  val max_iter = "5"
  val regularization = "2.0"
  val standardize = "false"
  val fit = "true"

  it should "work for Random Forest Classifier" in {
    val model = "Random Forest Classifier"
    val result = build_model(text, model, num_of_tree, max_depth, seed, smoothing, max_iter, regularization, standardize, fit)
    val accuracy  = result._1
    assert(accuracy > 0.6)
  }

  it should "work for Naive Bayesian Classifier" in {
    val model = "Naive Bayesian Classifier"
    val result = build_model(text, model, num_of_tree, max_depth, seed, smoothing, max_iter, regularization, standardize, fit)
    val accuracy  = result._1
    assert(accuracy > 0.7)
  }

  it should "work for SVM" in {
    val model = "Linear Support Vector Classification"
    val result = build_model(text, model, num_of_tree, max_depth, seed, smoothing, max_iter, regularization, standardize, fit)
    val accuracy  = result._1
    assert(accuracy > 0.7)
  }

}
