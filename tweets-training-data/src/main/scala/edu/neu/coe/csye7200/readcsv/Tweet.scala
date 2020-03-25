package edu.neu.coe.csye7200.readcsv

import scala.collection.mutable
import scala.io.Source
import scala.util.Try

case class Tweet(id: Int, keyword: String, location: String, text: String, target: Boolean)

object Tweet extends App {

  trait IngestibleTweet extends Ingestible[Tweet] {
    def fromString(w: String): Try[Tweet] = Try(apply(w.split(",")))
  }

  implicit object IngestibleTweet extends IngestibleTweet

  val ingester = new Ingest[Tweet]()
  if (args.length > 0) {
    val source = Source.fromFile(args.head)
    val tweets: Iterator[Try[Tweet]] =
      for (ty <- ingester(source)) yield
        for (t <- ty) yield t
    tweets foreach { _ foreach { println(_) } }
    source.close()
  }

  /**
    * Form a list from the elements explicitly specified (by position) from the given list
    *
    * @param list    a list of Strings
    * @param indices a variable number of index values for the desired elements
    * @return a list of Strings containing the specified elements in order
    */
  def elements(list: Seq[String], indices: Int*): List[String] = {
    val x = mutable.ListBuffer[String]()
    // Hint: form a new list which is consisted by the elements in list in position indices. Int* means array of Int.
    for (index <- indices) {
      x += list(index)
    }
    x.toList
  }

  /**
    * Alternative apply method for the Movie class
    *
    * @param ws a sequence of Strings
    * @return a Movie
    */
  def apply(ws: Seq[String]): Tweet = {
    // we ignore facenumber_in_poster since I have no idea what that means.
    val id = ws(0).toInt
    val keyword = ws(1)
    val location = ws(2)
    val text = ws(3)
    val target = if (ws(4).toInt == 1) true else false
    Tweet(id, keyword, location, text, target)
  }
}
