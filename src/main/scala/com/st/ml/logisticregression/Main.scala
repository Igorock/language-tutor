package com.st.ml.logisticregression

import org.apache.log4j.{Level, Logger}
import org.apache.log4j.lf5.LogLevel
import org.apache.spark.SparkConf
import org.apache.spark.ml.feature.{CountVectorizer, HashingTF, RegexTokenizer, Tokenizer}
import org.apache.spark.sql.SparkSession

object Main {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val conf = new SparkConf().setAppName("test-ml").setMaster("local[2]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._

    val sentenceData = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark, about Spark 23 as34"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat"),
      (2, "34 df4 end [d (df) (as mid sd) end."),
      (3, "2d3 r4")
    )).toDF("label", "sentence")

    val tokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setMinTokenLength(2)
      .setPattern("[\\d\\W]+")
      //.setPattern("(^| )[^ ]*[^A-Za-z ][^ ]*(?=$| )")
      //.setPattern("[^a-zA-Z]")
    val wordsData = tokenizer.transform(sentenceData)
    val hashingTF = new CountVectorizer()
      .setInputCol("words").setOutputCol("rawFeatures")

    val featurizedModel = hashingTF.fit(wordsData)
    val featurizedData = featurizedModel.transform(wordsData)

    featurizedData.show(false)

  }
}
