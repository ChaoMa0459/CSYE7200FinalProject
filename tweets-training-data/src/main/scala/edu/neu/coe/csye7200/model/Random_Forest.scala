package edu.neu.coe.csye7200.model

import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.functions._

class Random_Forest {
  //Read csv into dataframe
  val spark = org.apache.spark.sql.SparkSession.builder
    .master("local")
    .appName("Spark CSV Reader")
    .getOrCreate;
  val df_train = spark.read
    .format("csv")
    .option("header", "true")
    .option("mode", "DROPMALFORMED")
    .load("src/test/resources/train.csv")
  val df_test = spark.read
    .format("csv")
    .option("header", "true")
    .option("mode", "DROPMALFORMED")
    .load("src/test/resources/test.csv")

  df_train.show()
  df_test.show()

//Separate sentence in text column into words in df_train and df_test
  //train data
  val tokenizer_train = new Tokenizer().setInputCol("text").setOutputCol("words")
  val train_data_Tokenizer = new RegexTokenizer()
    .setInputCol("text")
    .setOutputCol("words")
    .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

  val countTokens_train = udf { (words: Seq[String]) => words.length }

  val tokenized_train = tokenizer_train.transform(df_train)
  tokenized_train.select("text", "words")
    .withColumn("tokens", countTokens_train(col("words"))).show(false)

  val train_data_Tokenized = train_data_Tokenizer.transform(df_train)
  train_data_Tokenized.select("text", "words")
    .withColumn("tokens", countTokens_train(col("words"))).show(false)

  //test data
  val tokenizer_test = new Tokenizer().setInputCol("text").setOutputCol("words")
  val test_data_Tokenizer = new RegexTokenizer()
    .setInputCol("text")
    .setOutputCol("words")
    .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

  val countTokens_test = udf { (words: Seq[String]) => words.length }

  val tokenized_test = tokenizer_test.transform(df_train)
  tokenized_test.select("text", "words")
    .withColumn("tokens", countTokens_test(col("words"))).show(false)

  val test_data_Tokenized = test_data_Tokenizer.transform(df_train)
  test_data_Tokenized.select("text", "words")
    .withColumn("tokens", countTokens_test(col("words"))).show(false)

//Romove the stop words in "text" column of train data and test data
  val remover = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filtered_words")

  remover.transform(train_data_Tokenized).show(false)
  remover.transform(test_data_Tokenized).show(false)
}
