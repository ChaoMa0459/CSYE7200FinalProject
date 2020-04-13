package edu.neu.coe.csye7200.model

import org.apache.spark.ml.feature.{HashingTF, IDF, RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._

import scala.collection.mutable



  object random_forest extends App {
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

    //Remove punctuate signs in text column
    val test_with_no_punct = df_train.collect().map(_.getString(3).replaceAll("""[\p{Punct}]""", "")).toSeq
    val rdd = spark.sparkContext.parallelize(test_with_no_punct)
    val rdd_train = df_train.rdd.zip(rdd).map(r => Row.fromSeq(r._1.toSeq ++ Seq(r._2)))
    val df_train_new = spark.createDataFrame(rdd_train, df_train.schema.add("new_text", StringType))

    //Separate sentence in text column into words in df_train and df_test
    //train data
    val tokenizer_train = new Tokenizer().setInputCol("new_text").setOutputCol("words")
    val train_data_Tokenizer = new RegexTokenizer()
      .setInputCol("new_text")
      .setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

    val countTokens_train = udf { (words: Seq[String]) => words.length }

    val tokenized_train = tokenizer_train.transform(df_train_new)
    tokenized_train.select("new_text", "words")
      .withColumn("tokens", countTokens_train(col("words"))).show(false)

    val train_data_Tokenized = train_data_Tokenizer.transform(df_train_new)
    train_data_Tokenized.select("new_text", "words")
      .withColumn("tokens", countTokens_train(col("words"))).show(false)

    //test data
    val tokenizer_test = new Tokenizer().setInputCol("text").setOutputCol("words")
    val test_data_Tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setPattern("\\W")

    val countTokens_test = udf { (words: Seq[String]) => words.length }

    val tokenized_test = tokenizer_test.transform(df_test.na.fill(Map("text" -> "")))
    tokenized_test.select("text", "words")
      .withColumn("tokens", countTokens_test(col("words"))).show(false)

    val test_data_Tokenized = test_data_Tokenizer.transform(df_test.na.fill(Map("text" -> "")))
    test_data_Tokenized.select("text", "words")
      .withColumn("tokens", countTokens_test(col("words"))).show(false)

    //Remove the stop words in "text" column of train data and test data
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered_words")

    remover.transform(train_data_Tokenized).show(false)
    remover.transform(test_data_Tokenized).show(false)

    val train_data = remover.transform(train_data_Tokenized).withColumn("tokens", countTokens_train(col("filtered_words")))

    val hashingTF = new HashingTF()
      .setInputCol("filtered_words").setOutputCol("rawFeatures").setNumFeatures(200)
    val featurizedData = hashingTF.transform(train_data)
    featurizedData.show(false)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.show(false)

    // word count
    // filter real tweets and count frequencies
    val real_train_data = rescaledData.filter("target == 1")
    var real_words_data: Seq[String] = Seq()
    real_train_data.foreach {
      row => {
        val filtered_words = row.toSeq(7)
        filtered_words match {
          case w: mutable.WrappedArray[String] => real_words_data ++= w
          case _ =>
        }
      }
    }

    val rdd_real_words = spark.sparkContext.parallelize(real_words_data)
    val real_words_counts = rdd_real_words
      .map(word => (word, 1))
      .reduceByKey(_ + _).sortBy(_._2, false)
    real_words_counts.take(20).foreach(println)

    // filter fake tweets and count frequencies
    val fake_train_data = rescaledData.filter("target == 0")
    var fake_words_data: Seq[String] = Seq()
    fake_train_data.foreach {
      row => {
        val filtered_words = row.toSeq(7)
        filtered_words match {
          case w: mutable.WrappedArray[String] => fake_words_data ++= w
          case _ =>
        }
      }
    }

    val rdd_fake_words = spark.sparkContext.parallelize(fake_words_data)
    val fake_words_counts = rdd_fake_words
      .map(word => (word, 1))
      .reduceByKey(_ + _).sortBy(_._2, false)
    fake_words_counts.take(20).foreach(println)

    spark.stop()
  }
