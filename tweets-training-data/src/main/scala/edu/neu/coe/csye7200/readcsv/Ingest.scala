package edu.neu.coe.csye7200.readcsv

import scala.io.Source
import scala.util.Try

class Ingest[T: Ingestible] extends (Source => Iterator[Try[T]]) {
  // parse the source file line by line in Seq[String]
  // drop the first line
  // then map it to Seq[Try[X]]
  // get the iterator of Try[X]
  def apply(source: Source): Iterator[Try[T]] = source.getLines.toSeq.drop(1).map(e => implicitly[Ingestible[T]].fromString(e)).iterator
}

trait Ingestible[X] {
  def fromString(w: String): Try[X]
}
