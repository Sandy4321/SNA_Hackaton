
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
    val numPartitionsGraph = 107
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
              .map(t => Friend(t.split(",")(0).toInt,int_mask_to_binary(t.split(",")(1).toInt)))
          }
          UserFriends2(user, friends)
        })
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



    //
    //  **** Random Sampling ****
    //

    val sample_filter_val = 1.0 / numPartitionsGraph * 2.5  // make sample size 20% larger than size of the partition
    // take 100% of ones and 25% of zeros
    val fractions: Map[AnyVal, Double] = Map(0 -> 0.25, 1.0 -> 1)

    val commonFriendsCounts = {
      sqlc
        //.read.parquet(commonFriendsPath + "/part_33")
        .read.parquet(commonFriendsPath + "/part_*")
        .map(t => (t.getAs[Int](0), t.getAs[Int](1)) -> t.getAs[Int](2))
        .leftOuterJoin(positives)
        .map(t => t._2._2.getOrElse(0) -> PairWithCommonFriends(t._1._1,t._1._2,t._2._1))
        .sampleByKey(withReplacement = false, fractions, 42)
        .map(t => {if (math.random<=sample_filter_val) 1 else 0} ->
          t._2)
        .filter(t => t._1==1)
        .map (t => t._2)
        .filter(pair => pair.person1 % 11 != 7 && pair.person2 % 11 != 7)
    }

    val count_aa = commonFriendsCounts.count()

    commonFriendsCounts.take(50).map(t => println(t))
    println (count_aa)
    //println (fractions[1])


    
     // **** Sceleton for parameter regularization ****
    


   // var roc_res = ArrayBuffer[RocVals]()

   // for( x <- Array(1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,0.1,0.5,1,5,10,15,30,60,100,150) ){
   //      roc_res += RocVals(x,0.4)
   //      println(x);
   //  }

   //  println (roc_res)







    // commonFriendsCounts.take(300).map(t => println(t))
    // println(commonFriendsCounts.count())





    //  *****
    //    A skeleton for stratified sampling
    //    Actually we can't expected this to be 100% correct
    //  ****

    // val data_values = {commonFriendsCounts
    //                .map (t => math.log(t.commonFriendsCount))
    //              }
                    
    // // an RDD[(K, V)] of any key value pairs
    // val max_val = data_values.max()
    // val min_val = data_values.min()

    // println(min_val)
    // println(max_val)

    // def get_group_num (x: Double, min_val: Double, max_val: Double, nbins: Int) = {
    //   val delta = (max_val - min_val)/ (nbins + 0.0)
    //   val gr_num = math.floor((x-min_val)/delta) + 1
    //   gr_num
    // }

    // val data = {commonFriendsCounts
    //                .map (t => get_group_num(math.log(t.commonFriendsCount),min_val,max_val+0.01,10) -> 1)
    //                .countByKey()
    //              }

    // data.map(t => println(t))







            
    }

}