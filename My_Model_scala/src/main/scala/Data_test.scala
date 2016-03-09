
import breeze.numerics.abs
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ArrayBuffer




object Data_test {

  def main22(args: Array[String]) {

    val sparkConf = new SparkConf()
      .setAppName("Baseline")
    val sc = new SparkContext(sparkConf)
    val sqlc = new SQLContext(sc)

    import sqlc.implicits._

    val dataDir = if (args.length == 1) args(0) else "./"

    val graphPath = dataDir + "trainGraph"
    val reversedGraphPath = dataDir + "trainSubReversedGraph"
    val commonFriendsPath = dataDir + "commonFriendsCountsPartitioned"
    val demographyPath = dataDir + "demography"
    val predictionPath = dataDir + "prediction"
    val modelPath = dataDir + "LogisticRegressionModel"
    val numPartitions = 200
    // val numPartitionsGraph = 107
    val numPartitionsGraph = 20


    println (demographyPath)
    val ageSex = {
      sc.textFile(demographyPath)
        .map(line => {
          val lineSplit = line.trim().split("\t")
          if (lineSplit(2) == "") {
            (lineSplit(0).toInt -> AgeSex(0, lineSplit(3).toInt))
          }
          else {
            (lineSplit(0).toInt -> AgeSex(lineSplit(2).toInt, lineSplit(3).toInt))
          }
        })
    }


    val CityReg = {
        sc.textFile(demographyPath)
          .map(line => {
            val lineSplit = line.trim().split("\t")
          if (lineSplit.length == 6) {
            (lineSplit(0).toInt -> UserCity(lineSplit(5).toInt, 0))
             }
          else {
            (lineSplit(0).toInt -> UserCity(lineSplit(5).toInt, lineSplit(6).toInt))
          }

          //if (lineSplit.length <3) {
            //lineSplit.length
          // if (lineSplit.length == 6 && lineSplit(2) == ""){

          //   (lineSplit(0).toInt -> AgeSex(0, lineSplit(3).toInt))
          // }
          // else {
          //   (lineSplit(0).toInt -> AgeSex(lineSplit(2).toInt, lineSplit(3).toInt))
          // }

        })

     } 

     CityReg.collect()
     CityReg.take(25).foreach(println)

     println ("\n \n Some important text \n \n")
    }

}