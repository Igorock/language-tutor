package com.st.ml.logisticregression

import java.io.File

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object TopicDetector {

  val logger = Logger.getLogger(TopicDetector.getClass)

  val ResourcesPath = "src/main/resources/"
  val TopicsPath = ResourcesPath + "topics/"
  val ModelPath = "/Users/igor.dziuba/models/topic-model"

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("LanguageTutor").setMaster("local[2]")
    implicit val spark = SparkSession.builder().config(conf).getOrCreate()
    import spark.implicits._

    val topicToPaths = new File(TopicsPath).listFiles()
      .map(file => removeExtension(file.getName) -> file.getPath)

    val topicData = topicToPaths.map { case (topic, path) =>
      spark.sparkContext.textFile(path)
        .toDF("text")
        .filter(length($"text") > 50)
        .withColumn("label", lit(topic))
    }.reduce(_ union _)
      .orderBy(rand())

    topicData.show()

    val stringIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(topicData)

    val regexTokenizer = new RegexTokenizer()
      .setMinTokenLength(3)
      .setPattern("[\\d\\W]+")
      .setInputCol("text")
      .setOutputCol("words")

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")

    val countVectorizer = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")

    val logisticRegression = new LogisticRegression()
      .setFeaturesCol("features")
      .setPredictionCol("prediction")
      .setLabelCol("indexedLabel")
      .setMaxIter(10)
      .setRegParam(0.5)

    val indexToString = new IndexToString()
      .setLabels(stringIndexer.labels)
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")

    val pipeline = new Pipeline()
      .setStages(Array(
        stringIndexer,
        regexTokenizer,
        stopWordsRemover,
        countVectorizer,
        idf,
        logisticRegression,
        indexToString
      ))

    val Array(trainData, testData) = topicData.randomSplit(Array(0.8, 0.2))

    val topicModel = pipeline.fit(trainData)

    val trainPredictions = topicModel.transform(trainData)
    val testPredictions = topicModel.transform(testData)

    testPredictions
      .withColumn("text", substring($"text", 0, 30))
      .select("text", "label", "predictedLabel", "prediction")
      .show(false)

    val evaluator = new MulticlassClassificationEvaluator()
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
      .setLabelCol("indexedLabel")

    val trainAccuracy = evaluator.evaluate(trainPredictions)
    val testAccuracy = evaluator.evaluate(testPredictions)

    logger.info("trainAccuracy: " + trainAccuracy)
    logger.info("testAccuracy: " + testAccuracy)

    val newTranslationsData = Seq(
      ("reason", "The surprising reason why NASA hasn't sent humans to Mars yet", "http://www.businessinsider.com/why-nasa-has-not-sent-humans-to-mars-2018-2"),
      ("related", "Despite having a population of only 40 million compared with the UK’s 65 million people, California’s gross domestic product of $2.7tn has overtaken the UK’s $2.6tn.", "https://en.wikipedia.org/wiki/Economy_of_California")
    ).toDF("word", "text", "source")
      .withColumn("text", concat($"text", lit(" "), $"source"))

    val finalPredictions = topicModel.transform(newTranslationsData)

    finalPredictions
      .select("word", "predictedLabel", "probability")
      .show(false)

  }

  private def removeExtension(name: String) =
    name.substring(0, name.lastIndexOf("."))

}
