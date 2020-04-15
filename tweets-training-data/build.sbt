name := "stop-word-remover"

organization := "com.ashrithgn.scala.tut"

version := "1.0"

scalaVersion := "2.11.10"

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.1" % "test",
  "org.apache.spark" %% "spark-core" % "2.4.5",
  "org.apache.spark" %% "spark-sql" % "2.4.5",
  "org.apache.spark" %% "spark-mllib" % "2.4.4" % "compile",
  "org.vegas-viz" %% "vegas" % "0.3.11",
  "org.vegas-viz" %% "vegas-spark" % "0.3.11",
  "org.scala-lang.modules" %% "scala-swing" % "2.0.1"
)