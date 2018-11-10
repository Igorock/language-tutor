package com.st.ml.logisticregression

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object ArticlesRecommender {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    val logger = Logger.getLogger(ArticlesRecommender.getClass)

    val conf = new SparkConf().setAppName("LanguageTutor").setMaster("local[2]")
    val spark = SparkSession.builder().config(conf).getOrCreate()
    val sc = spark.sparkContext
    import spark.implicits._

    val resourcesPath = "src/main/resources/"
    val topicsPath = resourcesPath + "topics/"

    val topics = Seq("machine-learning", "space", "music", "finance")

    val topicData = topics.map { topic =>
      sc.textFile(topicsPath + s"$topic.txt")
        .toDF("text")
        .withColumn("label", lit(topic))
    }.reduce(_ union _)
      //.filter(!col("text").rlike("^((?![a-zA-Z]).)*$"))
      .orderBy(rand()).cache()

    topicData.show(false)

    val stringIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(topicData)

    val indexed = stringIndexer.transform(topicData)

    indexed.show()

    val indexToString = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictionLabel")
      .setLabels(stringIndexer.labels)

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("words")
      .setMinTokenLength(2)
      .setPattern("[\\d\\W]+")

    val words = regexTokenizer.transform(indexed).drop("text")
    words.show()

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filtered")

    val filtered = stopWordsRemover.transform(words).drop("words").filter(size($"filtered") > 0)
    filtered.show()

    val hashingTF = new HashingTF()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")
      .setNumFeatures(2048)

    val hashed = hashingTF.transform(filtered)
    hashed.show(false)

    val countVectorizerTF = new CountVectorizer()
      .setInputCol("filtered").setOutputCol("rawFeatures")

    val featurizedModel = countVectorizerTF.fit(filtered)
    val featurizedData = featurizedModel.transform(filtered)

    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")
      .setMinDocFreq(0)

    val logisticRegression = new LogisticRegression()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setRegParam(0.5)

    val pipeline = new Pipeline()
      .setStages(Array(
        stringIndexer,
        regexTokenizer,
        stopWordsRemover,
        countVectorizerTF,
        idf,
        logisticRegression,
        indexToString
      ))

    val Array(trainData, testData) = topicData.randomSplit(Array(0.8, 0.2))
    val topicModel = pipeline.fit(trainData)

    val trainPredictions = topicModel.transform(trainData)
    val testPredictions = topicModel.transform(testData)

    testPredictions
      .select("text", "label", "predictionLabel", "probability")
      .show()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val trainAccuracy = evaluator.evaluate(trainPredictions)
    val testAccuracy = evaluator.evaluate(testPredictions)

    logger.info("trainAccuracy = " + trainAccuracy)
    logger.info("testAccuracy = " + testAccuracy)

    val extractWords = udf((link: String) => {
      link
        .replace("/$", "")
        .substring(link.lastIndexOf("/") + 1)
        .split("[\\d\\W_]+")
        .mkString(" ")
    })

    val userTranslations = spark.read.option("header","true").option("delimiter","|")
      .csv(resourcesPath + "userTranslations.csv")
      .withColumn("sourceBody", extractWords($"source"))
      .withColumn("text", concat($"text", lit(" "), $"sourceBody"))
      .cache

    userTranslations.show(false)

    val userTranslationsWithTopics = topicModel.transform(userTranslations)

    val newTranslationsData = Seq(
      ("reason", "The surprising reason why NASA hasn't sent humans to Mars yet", "http://www.businessinsider.com/why-nasa-has-not-sent-humans-to-mars-2018-2"),
      ("related", "Despite having a population of only 40 million compared with the UK’s 65 million people, California’s gross domestic product of $2.7tn has overtaken the UK’s $2.6tn.", "https://en.wikipedia.org/wiki/Economy_of_California")
    )

    val newTranslations = spark
      .createDataFrame(newTranslationsData)
      .toDF("word", "text", "source")
      .withColumn("text", concat($"text", $"source"))

    val newTranslationsWithTopics = topicModel.transform(newTranslations)

    newTranslationsWithTopics.show(false)

    userTranslationsWithTopics
      .select("word", "text", "predictionLabel", "probability")
      .filter($"word".isin(Seq("reason", "related"):_*))
      .show(false)

    val results = userTranslationsWithTopics.join(newTranslationsWithTopics, Seq("word", "predictionLabel"))
    results.show(false)


  }

}
