
//import breeze.numerics.abs
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
//import org.apache.spark.mllib.linalg.DenseVector
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.Row
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.feature.PCA


import collection.JavaConversions._



object Map_bin_test {

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





    // val graph = {
    //   sc.textFile(graphPath)
    //     .map(line => {
    //       val lineSplit = line.split("\t")
    //       val user = lineSplit(0).toInt
    //       val friends = {
    //         lineSplit(1)
    //           .replace("{(", "")
    //           .replace(")}", "")
    //           .split("\\),\\(")
    //           //.map(t => t.split(",")(0).toInt)
    //           .map(t => Friend(t.split(",")(0).toInt,t.split(",")(1).toInt))
    //       }
    //       UserFriends2(user, friends)
    //     })
    // }





    // //step 1.a from description
    // graph
    //   .filter(userFriends => userFriends.friends.length >= 8 && userFriends.friends.length <= 1000)
    //   .flatMap(userFriends => userFriends.friends.map(x => (x.user, (userFriends.user,x.mask_bit))))  // making new key
    //   .groupByKey(numPartitions)          // number of groups that will be created after partitioning
    //   .map(t => (t._1, t._2.toArray))
    //   .map(t => t._2)
    //   .filter(friends => friends.length >= 2 && friends.length <= 2000)
    //   .map(friends => new Tuple1(friends))      
    //   .toDF
    //   //.take(50).map(t => println(t))
    //   .write.parquet(reversedGraphPath + "_bin_map")



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


    def pairs_binary_count(binmap_person_1: Int, binmap_person_2: Int) = {

        def bin1 = int_mask_to_binary(binmap_person_1)
        def bin2 = int_mask_to_binary(binmap_person_2)
        val bin1_arr = ArrayBuffer.empty[Int]
        val bin2_arr = ArrayBuffer.empty[Int]
        var sum_bins =  Array.fill[Short](11*23)(0) //Vectors.zeros (11*23) // DenseVector.zeros[Short](12*23) //

        //  println(bin1)
        //  println(bin2)

        for (i<-1 until 22) {
            if (bin1(21-i) ==1)
                bin1_arr.append(i)
            if (bin2(21-i) ==1)
                bin2_arr.append(i)
            }

        if (bin1_arr.length ==0) bin1_arr.append(0)
        if (bin2_arr.length ==0) bin2_arr.append(0)

        // (bin1_arr)
        //println (bin2_arr)

        for (i<-0 until bin1_arr.length){
            for (j<-0 until bin2_arr.length){
               if (bin1_arr(i)>= bin2_arr(j)) 
                    sum_bins (bin1_arr(i) + 22 * bin2_arr(j) - (bin2_arr(j) * (bin2_arr(j)+1))/2) = 1
                else
                    sum_bins (bin2_arr(j) + 22 * bin1_arr(i) - (bin1_arr(i) * (bin1_arr(i)+1))/2) = 1
            }
        }

        
        //sum_bins :+ 1.toShort
        new DenseVector(sum_bins :+ 1.toShort)  // Last element added for counting purposes

    }





    val numPartitionsGraph: Int = 15

    // val k: Int = 0
    // sqlc.read.parquet(reversedGraphPath + "_bin_map")
    //               .map(t => t.getAs[Seq[Row]](0).map{case Row(k: Int, v: Int) => (k, v)}.toSeq)

    //               .map(t => generatePairs_v2(t, numPartitionsGraph, k))
    //               .flatMap(pair => pair.map(x => (x._1._1,x._2._1) -> (x._1._2,x._2._2)))
    //               .map(x => x._1-> pairs_binary_count(x._2._1,x._2._2))

    //               .reduceByKey((x, y) => x + y)
    //               .filter (x => x._2(253)>5)
    //               .map (x => x._1 -> x._2.slice(1, 253))
    //               .take(30).map(t => println(t))





    for (k <- 0 until numPartitionsGraph) {
        val commonFriendsCounts = {
            sqlc.read.parquet(reversedGraphPath + "_bin_map")
                  .map(t => t.getAs[Seq[Row]](0).map{case Row(k: Int, v: Int) => (k, v)}.toSeq)

                  .map(t => generatePairs_v2(t, numPartitionsGraph, k))
                  .flatMap(pair => pair.map(x => (x._1._1,x._2._1) -> (x._1._2,x._2._2)))
                  .map(x => x._1-> pairs_binary_count(x._2._1,x._2._2))

                  .reduceByKey((x, y) => x + y)
                  .filter (x => x._2(253)>5)
                  .map (x => x._1 -> x._2.slice(1, 253))
                  .map(x => new Tuple1(x._1._1, x._1._2, x._2.toArray.toSeq))
                  //.map (x => DenseVector(x._1._1,x._1._1) :+ x._2)
                  //.map (x => Tuple1(x._1._1,x._1._2))
                  //.take(50)
                  //.map (x => println(x))
        }


        commonFriendsCounts.toDF.repartition(4).write.parquet(commonFriendsPath + "_bin_map" + "/part_" + k)
    }


    // val commonFriendsCounts_bin_map = {
    //         sqlc
    //             .read.parquet(commonFriendsPath + "_bin_map" + "/part_*")
    //             .map(row =>
    //             {val key = row.getAs[Tuple(Int,Int)](0)


    //                 // {row(0)   match {
    //                 //                      case Row(k: Int, v:Int) => 
    //                 //                             {SimplePair(k,v)} //ArrayBuffer(k.toInt,v.toInt)
    //                 //                      case _ => 0}} 
    //             val value = row.getAs[Seq[Short]](1).toArray.map(_.toDouble)
    //             key -> value})
    //             //.map (t => t._1._1)

    //             .take(50)
    //             .map (t => println(t))
    //         }



    // def prepareData22( projected_bin_mask: RDD[((Int,Int), Array[Double])]) = {
    //     1
    // }

    //println(prepareData22 (commonFriendsCounts_bin_map))

    //val pca = new PCA(10).fit(commonFriendsCounts_bin_map.map(t => t._2))

    //val projected_bin_mask = commonFriendsCounts_bin_map.map(p => p._1 -> pca.transform(p._2))



    //////  val projected_bin_mask = commonFriendsCounts_bin_map.map (p => p._1.toString.toDouble)
    //////  projected_bin_mask.take(50).map (t => println(t))


    //pca.map(t => println(t))

    //val pcamat = commonFriendsCounts_bin_map.map(x => x._2).toDF.computePrincipalComponents(10)



// root
//  |-- _1: struct (nullable = true)
//  |    |-- _1: integer (nullable = true)
//  |    |-- _2: integer (nullable = true)
//  |-- _2: array (nullable = true)
//  |    |-- element: short (containsNull = true)


    }

}