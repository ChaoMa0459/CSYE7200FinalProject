package edu.neu.coe.csye7200.readcsv

import org.apache.spark.ml.feature.{HashingTF, IDF, IDFModel, RegexTokenizer, StopWordsRemover, Tokenizer}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.countDistinct

import scala.collection.mutable

object readCsv {
  val sparksession: SparkSession = org.apache.spark.sql.SparkSession.builder
    .master("local")
    .appName("Spark CSV Reader")
    .getOrCreate;

  sparksession.sparkContext.setLogLevel("ERROR")

  def readTrainData(): (DataFrame, Int) = {
    //Read csv into dataframe
    val df_train: DataFrame = sparksession.read
      .format("csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .load("src/main/resources/train.csv")
    //  val df_test: DataFrame = spark.read
    //    .format("csv")
    //    .option("header", "true")
    //    .option("mode", "DROPMALFORMED")
    //    .load("src/main/resources/test.csv")

//    df_train.show()
    //  df_test.show()

    //Remove punctuate signs in text column
    val test_with_no_punct: Seq[String] = df_train.collect().map(_.getString(3).replaceAll("https?://\\S+\\s?", "")
      .replaceAll("""[\p{Punct}]""", "")
      .replaceAll("Im", "i am")
      .replaceAll("Whats", "what is")
      .replaceAll("whats", "what is")
      .replaceAll("Ill", "i will")
      .replaceAll("theres", "there is")
      .replaceAll("Theres", "there is")
      .replaceAll("cant", "can not")).toSeq

    val rdd: RDD[String] = sparksession.sparkContext.parallelize(test_with_no_punct)
    val rdd_train: RDD[Row] = df_train.rdd.zip(rdd).map(r => Row.fromSeq(r._1.toSeq ++ Seq(r._2)))
    val df_train_new: DataFrame = sparksession.createDataFrame(rdd_train, df_train.schema.add("new_text", StringType))

    //Separate sentence in text column into words in df_train and df_test
    //train data
    val tokenizer_train: Tokenizer = new Tokenizer().setInputCol("new_text").setOutputCol("words")
    val train_data_Tokenizer: RegexTokenizer = new RegexTokenizer()
      .setInputCol("new_text")
      .setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

    val countTokens_train: UserDefinedFunction = udf { (words: Seq[String]) => words.length }

    val tokenized_train: DataFrame = tokenizer_train.transform(df_train_new)
    tokenized_train.select("new_text", "words")
      .withColumn("tokens", countTokens_train(col("words"))).show(false)

    val train_data_Tokenized: DataFrame = train_data_Tokenizer.transform(df_train_new)
    train_data_Tokenized.select("new_text", "words")
      .withColumn("tokens", countTokens_train(col("words"))).show(false)

    //Remove the stop words in "text" column of train data and test data
    val remover: StopWordsRemover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered_words")

    remover.transform(train_data_Tokenized).show(false)
    //  remover.transform(test_data_Tokenized).show(false)

    val train_data: DataFrame = remover.transform(train_data_Tokenized).withColumn("tokens", countTokens_train(col("filtered_words")))

    // TF
    val num_features = train_data.agg(countDistinct("filtered_words")).first().getLong(0)
    val num_features_int = num_features.toInt
    val hashingTF: HashingTF = new HashingTF()
      .setInputCol("filtered_words").setOutputCol("rawFeatures").setNumFeatures(num_features_int)
    val featuredData: DataFrame = hashingTF.transform(train_data)
    featuredData.show(false)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    // IDF
    val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel: IDFModel = idf.fit(featuredData)
    val rescaledData: DataFrame = idfModel.transform(featuredData)
    (rescaledData, num_features_int)
  }

  def readTestData(): (DataFrame, Int) = {
    //Read csv into dataframe
      val df_test: DataFrame = sparksession.read
        .format("csv")
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .load("src/main/resources/test.csv")

//    df_train.show()
//      df_test.show()

    //Remove punctuate signs in text column
    val test_with_no_punct: Seq[String] = df_test.collect().map(_.getString(3).replaceAll("https?://\\S+\\s?", "")
      .replaceAll("""[\p{Punct}]""", "")
      .replaceAll("Im", "i am")
      .replaceAll("Whats", "what is")
      .replaceAll("whats", "what is")
      .replaceAll("Ill", "i will")
      .replaceAll("theres", "there is")
      .replaceAll("Theres", "there is")
      .replaceAll("cant", "can not")).toSeq

    val rdd: RDD[String] = sparksession.sparkContext.parallelize(test_with_no_punct)
    val rdd_test: RDD[Row] = df_test.rdd.zip(rdd).map(r => Row.fromSeq(r._1.toSeq ++ Seq(r._2)))
    val df_test_new: DataFrame = sparksession.createDataFrame(rdd_test, df_test.schema.add("new_text", StringType))

    //Separate sentence in text column into words in df_train and df_test
    //test data
    val tokenizer_test: Tokenizer = new Tokenizer().setInputCol("new_text").setOutputCol("words")
    val test_data_Tokenizer: RegexTokenizer = new RegexTokenizer()
      .setInputCol("new_text")
      .setOutputCol("words")
      .setPattern("\\W") // alternatively .setPattern("\\w+").setGaps(false)

    val countTokens_test: UserDefinedFunction = udf { (words: Seq[String]) => words.length }

    val tokenized_test: DataFrame = tokenizer_test.transform(df_test_new)
    tokenized_test.select("new_text", "words")
      .withColumn("tokens", countTokens_test(col("words"))).show(false)

    val test_data_Tokenized: DataFrame = test_data_Tokenizer.transform(df_test_new)
    test_data_Tokenized.select("new_text", "words")
      .withColumn("tokens", countTokens_test(col("words"))).show(false)

    //Remove the stop words in "text" column of train data and test data
    val remover: StopWordsRemover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered_words")

    remover.transform(test_data_Tokenized).show(false)
    //  remover.transform(test_data_Tokenized).show(false)

    val test_data: DataFrame = remover.transform(test_data_Tokenized).withColumn("tokens", countTokens_test(col("filtered_words")))

    // TF
    val num_features_test = test_data.agg(countDistinct("filtered_words")).first().getLong(0)
    val num_features_test_int = num_features_test.toInt
    val hashingTF: HashingTF = new HashingTF()
      .setInputCol("filtered_words").setOutputCol("rawFeatures").setNumFeatures(num_features_test_int)
    val featuredData: DataFrame = hashingTF.transform(test_data)
    featuredData.show(false)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    // IDF
    val idf: IDF = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel: IDFModel = idf.fit(featuredData)
    val rescaledData: DataFrame = idfModel.transform(featuredData)
    (rescaledData, num_features_test_int)
  }

}

