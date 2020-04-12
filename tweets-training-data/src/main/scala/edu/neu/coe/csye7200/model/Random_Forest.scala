package edu.neu.coe.csye7200.model

import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Tokenizer, HashingTF, IDF}
import org.apache.spark.sql.types.{StructType,StructField,StringType,IntegerType};
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._



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

    val train_data = remover.transform(train_data_Tokenized).withColumn("tokens", countTokens_test(col("filtered_words")))

//    val sentenceData = spark.createDataFrame(Seq(
//      (0.0, "Hi I heard about Spark"),
//      (0.0, "I wish Java could use case classes"),
//      (1.0, "Logistic regression models are neat")
//    )).toDF("label", "sentence")
//
//    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
//    val wordsData = tokenizer.transform(sentenceData)
//
//    wordsData.show(false)

    val hashingTF = new HashingTF()
      .setInputCol("filtered_words").setOutputCol("rawFeatures").setNumFeatures(100)

    val featurizedData = hashingTF.transform(train_data)

    // featurizedData is ok
    featurizedData.show(false)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    // TODO there is an NPE at line 93.
    val idfModel = idf.fit(featurizedData)

//     val rescaledData = idfModel.transform(featurizedData)
//
//     rescaledData.show(false)
//
//     rescaledData.select("label", "features").show(false)

    spark.stop()
  }
