
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


    def int_mask_to_binary(k: Int) = {
            val t = k.toBinaryString
            //String.format("%0"+(20-t.length())+"d%s",0,t)
            val val22 = "%020d".format(0) + k.toBinaryString
            val22.takeRight(21).map(_.toString().toInt)
        
        }



    val graph = {
      sc.textFile(graphPath)
        .map(line => {
          val lineSplit = line.split("\t")
          val user = lineSplit(0).toInt
          val friends = {
            lineSplit(1)
              .replace("{(", "")
              .replace(")}", "")
              .split("\\),\\(")
              //.map(t => t.split(",")(0).toInt)
              .map(t => Friend(t.split(",")(0).toInt,int_mask_to_binary(t.split(",")(1).toInt)))
          }
          UserFriends2(user, friends)
        })
    }


    val friends_count = graph.map (t => t.user -> t.friends.length)

    friends_count.take(20).map(t => println(t))

    // val xx = "%021d".format(0).takeRight(21).map(_.toString().toInt)
    // val xx2 = Vectors.dense(1,2,3).toArray.deep
    // val tt = xx2.union(xx)

    // //val xx33 = Vectors.dense(Array(100, 1, 2))
    // //val xx3 = Vectors.dense("%021d".format(0).takeRight(21).map(_.toInt))

    // println (xx  )
    // println (xx2)
    // println (Vectors.dense(tt.toArray.map({l => l.toString().toDouble})))





            
    }

}