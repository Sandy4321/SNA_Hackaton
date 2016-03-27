
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





object CommonFriendsCountWithIds {

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



    // //
    // // Inverting and saving friends graph in specific way: friends -> common_friend_ID
    // //

    graph
          .filter(userFriends => userFriends.friends.length >= 8 && userFriends.friends.length <= 1000)
           // making change from "user -> friend" to "friend -> user"
          .flatMap(userFriends => userFriends.friends.map(x => (x.user, userFriends.user))) 
          
          .groupByKey(numPartitions)          // number of groups that will be created after partitioning
          .map(t => UserFriends(t._1, t._2.toArray))
          .map(userFriends => (userFriends.friends.sorted, userFriends.user))
          .filter(friends => friends._1.length >= 2 && friends._1.length <= 2000)
          .map(t => UserFriendsReversed (t._1,t._2))
          .toDF
          .write.parquet(reversedGraphPath + "_userID")




    def generatePairs(pplWithCommonFriends: Seq[Int], numPartitions: Int, k: Int) = {
      val pairs = ArrayBuffer.empty[(Int, Int)]
      for (i <- 0 until pplWithCommonFriends.length) {
        if (pplWithCommonFriends(i) % numPartitions == k) {
          for (j <- i + 1 until pplWithCommonFriends.length) {
            pairs.append((pplWithCommonFriends(i), pplWithCommonFriends(j)))
          }
        }
      }
      pairs
    }



    val usersBC = sc.broadcast(graph.map(userFriends => userFriends.user).collect().toSet)

    //
    // Create all pairs from usersBC like : (key = (user, friend), val = 1) 
    // Here friend.id > user.id as it is in commonFriendCount data. 
    // 
    // positives - positive class
    //
    val positives = {
      graph
        .flatMap(
          userFriends => userFriends.friends
            .filter(x => (usersBC.value.contains(x.user) && x.user > userFriends.user))
            .map(x => (userFriends.user, x.user) -> 1.0)
        )
    }



    val friendscountBC = sc.broadcast(graph.map (t => t.user -> t.friends.length).collectAsMap())

    //
    // Calculate commonFriendsCounts with additional relative measures:
    // - if two people are connected
    // - common friends div max friends
    // - common friends div min friends
    //

    // structure: 
    // (person1, person2) -> (common friends, are p1 and p2 friends, 
    //                        common friends/max_friends, common friends/min_friends)

    val commonFriendsCounts_addit = {
      sqlc
        .read.parquet(commonFriendsPath + "/part_*")
        .map(t => (t.getAs[Int](0), t.getAs[Int](1)) -> t.getAs[Int](2))
        .leftOuterJoin(positives)
        .map (t => t._1 -> (t._2._1, t._2._2.getOrElse(0.0) ))
        .map (t => t._1 -> (t._2._1,t._2._2, 
                                friendscountBC.value.getOrElse(t._1._1,0),
                                friendscountBC.value.getOrElse(t._1._2,0)  ))

        .map (t => t._1 -> (t._2._1, t._2._2, t._2._1.toDouble/math.max(t._2._3,t._2._4),
                                              t._2._1.toDouble/math.min(t._2._3,t._2._4)))
    }



    //
    //  Generating pairs of common friends for future analysis
    //

    val k = 1
    val commonFriendsCounts_with_ids = {
                sqlc.read.parquet(reversedGraphPath + "_userID")
//                         .map(t =>  generatePairs(t.getAs[Seq[Int]](0), numPartitionsGraph, k) -> t.getAs[Int](1))
                         .map(t =>  generatePairs(t.getAs[Seq[Int]](0),1,0) -> t.getAs[Int](1))

                        // making pairs: ((person1_id, person2_id), common_friend_id)
                         .flatMap(pair => pair._1.map(x => x -> pair._2))
                         .groupByKey()
                         .map (t => t._1 -> t._2.toSeq.sorted)
                         .map (t => t._1 -> generatePairs(t._2.toSeq,1,0))
                         .filter (t => t._2.length>0)
                         .flatMap (pairs => pairs._2.map(x => x -> pairs._1))
                         .leftOuterJoin(commonFriendsCounts_addit)
                         .filter (t => t._2._2 != None)
                         .map (t => t._2._1 -> t._2._2.toVector(0))
                         .reduceByKey((a,b) => (
                                                // Append constructions of lists and quantiles
                                                math.max(a._1,b._1),
                                                a._2 + b._2,
                                                math.max(a._3,b._3),
                                                math.max(a._4,b._4)
                                                ))
                         // probably bettter to save in parquet?
                         //.map (t => CoomonFriendswithFriendsStats(t._1._1,t._1._2,t._2._1,t._2._2,t._2._3,t._2._4))
                         //.take(50)
                         //.map(println)
            }


    commonFriendsCounts_with_ids.repartition(4).saveAsTextFile(commonFriendsPath + "_with_ids" + "/part_" + k,  classOf[GzipCodec])
            
    }

}