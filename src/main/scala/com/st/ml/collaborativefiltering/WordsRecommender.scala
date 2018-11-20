package com.st.ml.collaborativefiltering

import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkConf
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

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
    import spark.implicits._

    val userWordData = spark.read.option("header", "true").csv(WordsPath + "user_word_count.csv")
    val rawWordData = spark.read.option("header", "true").csv(WordsPath + "words.csv")
    val wordAlias = spark.read.option("header", "true").csv(WordsPath + "word_alias.csv")
    val badWords = spark.read.csv(WordsPath + "full-list-of-bad-words.csv").toDF("name")
    val allEnglishWords = spark.read.textFile(WordsPath + "en-words-20k.txt").toDF("name")


    val wordData = rawWordData.join(badWords, Seq("name"), "leftanti")
      .join(allEnglishWords, Seq("name"))
      .filter(col("name") rlike "\\b[^\\d\\W]{3,}+\\b")

    wordData.show(false)

    val wordAliasMap = wordAlias.collect().map { line =>
      (line.getString(0).toInt, line.getString(1).toInt)}.toMap

    val bWordAlias = spark.sparkContext.broadcast(wordAliasMap)

    val groupedWordData = userWordData.map { line =>
      val (userID, wordID, count) = (line.getString(0).toInt, line.getString(1).toInt, line.getString(2).toInt)
      val finalWordID = bWordAlias.value.getOrElse(wordID, wordID)
      (userID, finalWordID, count)
    }.toDF("user", "word", "count")

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
      fit(groupedWordData)

    groupedWordData.unpersist()

    val userID = 8
    val userData = groupedWordData.where(s"user == $userID")

    userData.join(wordData, wordData("id") <=> groupedWordData("word"), "left").show()

    val toRecommend = model.itemFactors.
      select($"id".as("word")).
      withColumn("user", lit(userID))

    val topRecommendations = model.transform(toRecommend).
      select("word", "prediction").
      orderBy($"prediction".desc).
      limit(20)

    val recommendedWordIDs = topRecommendations.select("word").join(userData, Seq("word"), "leftanti").toDF("id")

    val finalTopRecommendations = wordData.join(recommendedWordIDs, "id")

    finalTopRecommendations.show(false)


  }

}
