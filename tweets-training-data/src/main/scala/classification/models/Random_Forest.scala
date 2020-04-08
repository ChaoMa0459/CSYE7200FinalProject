package classification.models

import org.apache.spark.sql.types.{StructType,StructField,StringType,IntegerType};
import org.apache.spark.sql.Row;

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
}
