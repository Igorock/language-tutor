package com.st.ml.collaborativefiltering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType

import scala.util.Random

object WordsRecommender {

  val logger = Logger.getLogger(WordsRecommender.getClass)

  val ResourcesPath = "src/main/resources/"
  val WordsPath = ResourcesPath + "words/"

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("LanguageTutor").setMaster("local[2]")
    conf.set("spark.sql.crossJoin.enabled", "true")
    implicit val spark = SparkSession.builder().config(conf).getOrCreate()
    import org.apache.spark.sql.Encoders
    import spark.implicits._

    val rawUserWordData = spark.read
      .schema(Encoders.product[(Int,Int,Int)].schema)
      .option("header", "true")
      .csv(WordsPath + "user_word_count.csv")
      .toDF("user", "word", "count")

    val rawWordData = spark.read
      .option("header", "true")
      .csv(WordsPath + "words.csv")

    val badWords = spark.read
      .csv(WordsPath + "full-list-of-bad-words.csv")
      .toDF("name")

    val allEnglishWords = spark.read
      .textFile(WordsPath + "en-words-20k.txt")
      .toDF("name")

    logger.info("Filtering word data...")

    val wordData = rawWordData
      .join(badWords, Seq("name"), "leftanti")
      .join(allEnglishWords, Seq("name"))
      .filter(col("name") rlike "\\b[^\\d\\W]{3,}+\\b")

    wordData.show(false)

    logger.info("Filtering userWord data...")

    val userWordData = rawUserWordData
      .join(wordData, wordData("id") <=> rawUserWordData("word"))
      .select("user", "word", "count")

    userWordData.show(false)

    logger.info("Building model...")

    val model = new ALS().
      setSeed(Random.nextLong()).
      setImplicitPrefs(true).
      setRank(10).
      setRegParam(1.0).
      setAlpha(80.0).
      setMaxIter(20).
      setNonnegative(true).
      setUserCol("user").
      setItemCol("word").
      setRatingCol("count").
      setPredictionCol("prediction").
      fit(userWordData)

    userWordData.unpersist()

    val userID = 8
    val userData = userWordData.where(s"user == $userID")

    val toRecommend = userData.
      withColumn("word", $"word".cast(IntegerType))

    toRecommend.show(false)

    logger.info(s"Getting recommendations for user ID = $userID ...")

    val topRecommendations = model.recommendForUserSubset(toRecommend, 20)
      .select(explode($"recommendations").alias("recommendation"))
      .select("recommendation.*")

    topRecommendations.show(false)

    val recommendedWordIDs = topRecommendations
      .withColumn("id", $"word")

    logger.info(s"Replacing word IDs with word text values...")

    val finalTopRecommendations = wordData
      .join(recommendedWordIDs, "id")
      .select($"name".alias("recommended_word"))

    finalTopRecommendations.show(false)

  }

}
