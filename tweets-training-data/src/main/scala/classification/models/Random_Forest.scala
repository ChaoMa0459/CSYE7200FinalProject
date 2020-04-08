package classification.models

import org.apache.spark.sql.types.{StructType,StructField,StringType,IntegerType};
import org.apache.spark.sql.Row;
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

class Random_Forest {
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
  val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
  val regexTokenizer = new RegexTokenizer()
    .setInputCol("text")
    .setOutputCol("words")
    .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

  val countTokens = udf { (words: Seq[String]) => words.length }

  val tokenized = tokenizer.transform(df_train)
  tokenized.select("text", "words")
    .withColumn("tokens", countTokens(col("words"))).show(false)

  val regexTokenized = regexTokenizer.transform(df_train)
  regexTokenized.select("text", "words")
    .withColumn("tokens", countTokens(col("words"))).show(false)

}
