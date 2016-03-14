
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
import org.apache.spark.sql.Row
import scala.collection.mutable.ArrayBuffer






object Data_test {

  def main(args: Array[String]) {

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
    val modelDataPath = dataDir + "Data_Model"
    val modelPath = dataDir + "LogisticRegressionModel"
    val numPartitions = 200
    //val numPartitionsGraph = 107
    //val numPartitionsGraph = 20


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
              .map(t => Friend(t.split(",")(0).toInt,t.split(",")(1).toInt))
          }
          UserFriends2(user, friends)
        })
    }





    // step 1.a from description
    graph
      .filter(userFriends => userFriends.friends.length >= 8 && userFriends.friends.length <= 1000)
      .flatMap(userFriends => userFriends.friends.map(x => (x.user, (userFriends.user,x.mask_bit))))  // making new key
      .groupByKey(numPartitions)          // number of groups that will be created after partitioning
      .map(t => (t._1, t._2.toArray))
      .map(t => t._2)
      .filter(friends => friends.length >= 2 && friends.length <= 2000)
      .map(friends => new Tuple1(friends))      
      .toDF
      //.take(50).map(t => println(t))
      .write.parquet(reversedGraphPath + "_bin_map")



    // Generate pairs with binary codes
    def generatePairs_v2(pplWithCommonFriends: Seq[(Int,Int)], numPartitions: Int, k: Int) = {
      val pairs = ArrayBuffer.empty[((Int,Int), (Int,Int))]
      for (i <- 0 until pplWithCommonFriends.length) {
        if (pplWithCommonFriends(i)._1 % numPartitions == k) {
          for (j <- i + 1 until pplWithCommonFriends.length) {
            pairs.append((pplWithCommonFriends(i), pplWithCommonFriends(j)))
          }
        }
      }
      pairs
    }


    val numPartitionsGraph: Int = 1
    val k: Int = 0
    sqlc.read.parquet(reversedGraphPath + "_bin_map")
          .map(t => t.getAs[Seq[Row]](0).map{case Row(k: Int, v: Int) => (k, v)}.toSeq)
          //.map(t => t.getAs[Seq[(Int,Int)]](0).toSeq)
          //.take(50).map(t => println(t))
          .map(t => generatePairs_v2(t, numPartitionsGraph, k))
          .flatMap(pair => pair.map(x => (x._1._1,x._2._1) -> (x._1._2,x._2._2)))
          .groupByKey() 
          .map(t => (t._1, t._2.toSeq))
          .take(500).map(t => println(t))

          // .flatMap(pair => pair.map(x => x -> 1))
          // .reduceByKey((x, y) => x + y)
          // .map(t => PairWithCommonFriends(t._1._1, t._1._2, t._2))
          // .filter(pair => pair.commonFriendsCount > 8)



            
    }

}