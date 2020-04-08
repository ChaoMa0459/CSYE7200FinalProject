package classification.models

import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.{MLUtils, }

class Random_Forest {

  val schema = StructType(
    StructField("keyword", StringType, nullable = true) ::
    StructField("location", StringType, nullable = true) ::
    StructField("result", StringType, nullable = false) ::
    Nil)

  val creditDf = spark.read.format("csv").option("header", value = true).option("delimiter", ",").option("mode", "DROPMALFORMED")
     .schema(schema)
     .load(getClass.getResource("/Users/sunyan/Scala_Project/CSYE7200FinalProject/tweets-training-data/src/test/resources/test.csv").getPath)
     .cache()
  creditDf.printSchema()
}
