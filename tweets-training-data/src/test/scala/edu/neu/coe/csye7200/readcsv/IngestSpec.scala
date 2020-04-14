package edu.neu.coe.csye7200.readcsv

import org.scalatest.{FlatSpec, Matchers}

import scala.io.{Codec, Source}
import scala.util._

class IngestSpec extends FlatSpec with Matchers {

  behavior of "ingest"

  it should "work for Int" in {
    trait IngestibleInt extends Ingestible[Int] {
      def fromString(w: String): Try[Int] = Try(w.toInt)
    }
    implicit object IngestibleInt extends IngestibleInt
    val source = Source.fromChars(Array('x', '\n', '4', '2'))
    val ingester = new Ingest[Int]()
    val xys = ingester(source).toSeq
    // check that xys has exactly one element, consisting of Success(42) -- 10 points
    // TO BE IMPLEMENTED
    xys should have size 1
    xys.head should matchPattern {
      case Success(42) =>
    }
  }

  it should "work for training data" in {
    implicit val codec: Codec = Codec("UTF-8")
    // NOTE that you expect to see a number of exceptions thrown. That's OK. We expect that some lines will not parse correctly.
    Try(Source.fromResource("train.csv")) match {
      case Success(source) =>
        val ingester = new Ingest[Tweet]()
        // get Seq[Try[Tweet]] using the iterator ingester(source)
        val tys: Seq[Try[Tweet]] = (for (ty <- ingester(source)) yield ty.transform(
          { t => Success(t) }, { e => System.err.println(e); ty }
        )).toSeq
        // val tos: Seq[Option[Tweet]] = for (ty <- tys) yield for (t <- ty.toOption) yield t
        val tos: Seq[Option[Tweet]] = for (ty <- tys) yield ty.toOption
        val ts = tos.flatten
        ts.size shouldBe 5421
        ts foreach { println(_) }
        source.close()
      case Failure(x) =>
        fail(x)
    }
  }

}
