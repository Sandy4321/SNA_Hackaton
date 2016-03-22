/**
  * Baseline for hackaton


  5. build project sbt package
  ... 
  ...
  9. send jar you made in step 5 to spark (configuration is given for 4 cores)
  ...
   spark-1.6.0/bin/spark-submit --class "Baseline" --master local[8] --driver-memory 8G My_Model_scala/target/scala-2.10/baseline_2.10-1.0.jar Data_short/


   */

import breeze.linalg.DenseVector
import breeze.numerics.abs
import org.apache.hadoop.io.compress.GzipCodec
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.sql.Row

import org.apache.spark.sql.Row
import scala.collection.mutable.ArrayBuffer
import org.apache.spark.mllib.feature.PCA


object Baseline_full_v1_testing_iter {

  def main(args: Array[String]) {

    val sparkConf = new SparkConf()
      .setAppName("Baseline")
      .set("spark.local.dir","/mnt/dev1/spark-temp")

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
    val numPartitionsGraph = 107

    // val numPartitionsGraph = 10
    val PCAsize = 47      //  best size is 51
    val create_counters = false


    //
    // https://habrahabr.ru/company/odnoklassniki/blog/277527/
    //

    // step 1.0 read graph, flat and reverse it
    //
    // sc - spark context type
    // textFile support ".gz" files
    //
    // input data type:
    // 102416 {(5362439,0), (17772,0),(674295,0), ... }
    // 2736 {(2542,0),(4570,0),(25832,0),(43782,0), ... }


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

    
    if (create_counters) {

        // step 1.a from description
        graph
          .filter(userFriends => userFriends.friends.length >= 8 && userFriends.friends.length <= 1000)
          .flatMap(userFriends => userFriends.friends.map(x => (x.user, userFriends.user)))  // making new key
          .groupByKey(numPartitions)          // number of groups that will be created after partitioning
          .map(t => UserFriends(t._1, t._2.toArray))
          .map(userFriends => userFriends.friends.sorted)
          .filter(friends => friends.length >= 2 && friends.length <= 2000)
          .map(friends => new Tuple1(friends))
          .toDF
          .write.parquet(reversedGraphPath)
    }


    // for each pair of plp count the amount of their common friends
    // amount of shared friends for pair (A, B) and for pair (B, A) is the same
    // so order pair: A < B and count common friends for pairs unique up to permutation
    // step 1.b

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


    if (create_counters) {

                  for (k <- 0 until numPartitionsGraph) {
                    val commonFriendsCounts = {
                      sqlc.read.parquet(reversedGraphPath)
                        .map(t => generatePairs(t.getAs[Seq[Int]](0), numPartitionsGraph, k))
                        .flatMap(pair => pair.map(x => x -> 1))
                        .reduceByKey((x, y) => x + y)
                        .map(t => PairWithCommonFriends(t._1._1, t._1._2, t._2))
                        .filter(pair => pair.commonFriendsCount > 8)
                    }

                    commonFriendsCounts.toDF.repartition(4).write.parquet(commonFriendsPath + "/part_" + k)
                  }




              // Generating mask PCA variables
              // step 2
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
                            .map(x => PairWithCommonFriendsAndFriendMask(x._1._1, x._1._2, x._2.toArray))
                  }
                   commonFriendsCounts.toDF.repartition(4).write.parquet(commonFriendsPath + "_bin_map" + "/part_" + k)
               }
   }


  val commonFriendsCounts_bin_mask = {
            sqlc
                .read.parquet(commonFriendsPath + "_bin_map" + "/part_*")
                .map(row =>
                    (row.getAs[Int](0),
                    row.getAs[Int](1)) ->
                    Vectors.dense(row.getAs[Seq[Short]](2).toArray.map(_.toDouble)))
            }







    // step 3
    //
    // list of all users
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

    val sample_filter_val = 1.0 / numPartitionsGraph * 10  // make sample size 20% larger than size of the partition
    // take 100% of ones and 25% of zeros
    val fractions: Map[AnyVal, Double] = Map(0 -> 0.20, 1.0 -> 1)

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



    // step 4
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

        })

     } 

    val ageSexBC = sc.broadcast(ageSex.collectAsMap())
    val cityRegBC = sc.broadcast(CityReg.collectAsMap())
    val friendscountBC = sc.broadcast(graph.map (t => t.user -> t.friends.length).collectAsMap())


    // Bit mask for type of friend
    val friend_masks = {
          graph
            .flatMap(
              userFriends => userFriends.friends
                .filter(x => (usersBC.value.contains(x.user) && x.user > userFriends.user))
                .map(x => (userFriends.user, x.user) -> 
                    {val k: Seq[Int] = int_mask_to_binary(x.mask_bit); 
                     k})
            )
        }



    // step 5
    def prepareData(
                     commonFriendsCounts: RDD[PairWithCommonFriends],
                     positives: RDD[((Int, Int), Double)],
                     friend_masks: RDD[((Int, Int), Seq[Int])],
                     projected_bin_mask: RDD[((Int, Int), Array[Double])],
                     common_bin_mask_counts: RDD[((Int, Int), Array[Double])],
                     ageSexBC:  Broadcast[scala.collection.Map[Int, AgeSex ]],
                     cityRegBC: Broadcast[scala.collection.Map[Int, UserCity]],
                     friendscountBC: Broadcast[scala.collection.Map[Int, Int]]) = {

      val zero_fr_masks_lst = "%021d".format(0).takeRight(21).map(_.toString().toInt)
      val zero_masks_pca = "%0250d".format(0).takeRight(PCAsize).map(_.toString().toInt).toArray
      val zero_masks_common_bin_mask = "%025d".format(0).takeRight(22).map(_.toString().toInt).toArray


      commonFriendsCounts
        .map(pair => (pair.person1, pair.person2) -> (Vectors.dense(
          pair.commonFriendsCount.toDouble,


          if (friendscountBC.value.getOrElse(pair.person1,0) != 0)
                         pair.commonFriendsCount.toDouble/friendscountBC.value.getOrElse(pair.person1,0) else 0,


          if (friendscountBC.value.getOrElse(pair.person2,0) != 0)
                         pair.commonFriendsCount.toDouble/friendscountBC.value.getOrElse(pair.person2,0) else 0,


          if (friendscountBC.value.getOrElse(pair.person1,0) != 0 && friendscountBC.value.getOrElse(pair.person2,0) != 0)
                         pair.commonFriendsCount.toDouble/math.max(friendscountBC.value.getOrElse(pair.person1,0),friendscountBC.value.getOrElse(pair.person1,0)) else 0,


          pair.commonFriendsCount.toDouble/friendscountBC.value.getOrElse(pair.person2,999999),
          abs(ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0)).age - ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0)).age).toDouble,
          // sex
          if (ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0)).sex == ageSexBC.value.getOrElse(pair.person2, AgeSex(0, 0)).sex && 
              ageSexBC.value.getOrElse(pair.person1, AgeSex(0, 0)).sex != 0) 1.0 else 0.0
          // city of residence
          ,if (cityRegBC.value.getOrElse(pair.person1, UserCity(-1, -1)).city == cityRegBC.value.getOrElse(pair.person2, UserCity(-1, -1)).city &&
               cityRegBC.value.getOrElse(pair.person1, UserCity(-1, -1)).city != -1) 1.0 else 0.0
          // city of active
          ,if (cityRegBC.value.getOrElse(pair.person1, UserCity(-1, -1)).city_active == cityRegBC.value.getOrElse(pair.person2, UserCity(-1, -1)).city_active &&
               cityRegBC.value.getOrElse(pair.person1, UserCity(-1, -1)).city_active != -1) 1.0 else 0.0
          ))

        )
        // .leftOuterJoin(friend_masks)
        // .map(x => x._1 -> (x._2._1.toArray.deep.union(x._2._2.getOrElse(zero_fr_masks_lst))))  // join with friend_masks
        // .map(x => x._1 -> (Vectors.dense(x._2.toArray.map({l => l.toString().toDouble}))))  // convert back to vector_dense
  
        .leftOuterJoin(projected_bin_mask)
        .map(x => x._1 -> (x._2._1.toArray.deep.union(
                            x._2._2.getOrElse(zero_masks_pca))))  // join with friend_masks
        .map(x => x._1 -> (Vectors.dense(x._2.toArray.map({l => l.toString().toDouble}))))  // convert back to vector_dense


        .leftOuterJoin(common_bin_mask_counts)
        .map(x => x._1 -> (x._2._1.toArray.deep.union(
                            x._2._2.getOrElse(zero_masks_common_bin_mask) )))  // join with friend_masks

        //                  x._2._2.getOrElse(zero_masks_common_bin_mask).map (l => l.toString().toDouble / math.max(1,x._2._1(0)) ))))  // join with friend_masks
        .map(x => x._1 -> (Vectors.dense(x._2.toArray.map({l => l.toString().toDouble}))))  // convert back to vector_dense

        .leftOuterJoin(positives)
        
    }



    def get_common_counts (all_common_frieds: Array[Short]) = {

        var sum_bins =  Array.fill[Double](22)(0)
        for (i <- 0 until 21)
            sum_bins(i) = all_common_frieds(i + 22 * i - (i * (i+1))/2)
        sum_bins
    } 


    // Count common friends by their mask
    val common_bin_mask_counts = commonFriendsCounts_bin_mask.map(t => t._1 -> get_common_counts(t._2.toArray.map(l => l.toShort)))


    val pca = new PCA(PCAsize).fit(commonFriendsCounts_bin_mask.map(t => t._2))
    val projected_bin_mask = commonFriendsCounts_bin_mask.map(p => p._1 -> pca.transform(p._2).toArray)

    //
    // if point class is not positive than we make it zero
    //
    val data = {
      prepareData(commonFriendsCounts, positives, friend_masks, projected_bin_mask, common_bin_mask_counts, ageSexBC, cityRegBC, friendscountBC)
        .map(t => LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
    }





    //  split data into training (10%) and validation (90%)
    //  step 6
    val splits = data.randomSplit(Array(0.3, 0.7), seed = 11L)
    val training_ns = splits(0)
    val validation_ns = splits(1)

    // Scalling data
    val scaler1 = new StandardScaler().fit(training_ns.map(x => x.features))
    val training = training_ns.map(x => LabeledPoint(x.label, scaler1.transform(x.features))).cache()
    val validation = validation_ns.map(x => LabeledPoint(x.label, scaler1.transform(x.features)))


    val y_positive = training.filter(x => x.label==1).count()
    val y_negative = training.filter(x => x.label==0).count()


    // run training algorithm to build the model


    val step_iter = 3000
    // https://www.kaggle.com/rootua/avito-context-ad-clicks/apache-spark-scala-logistic-regression/run/27034
    val model_not_trained =  new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true)
    model_not_trained.optimizer.setNumIterations(step_iter)
    val model = model_not_trained.run(training)


    model.clearThreshold()
    //model.save(sc, modelPath)

    val predictionAndLabels = {
      validation.map { case LabeledPoint(label, features) =>
        val prediction = model.predict(features)
        (prediction, label)
      }
    }

    // estimate model quality
    @transient val metricsLogReg = new BinaryClassificationMetrics(predictionAndLabels, 100)
    val threshold = metricsLogReg.fMeasureByThreshold(2.0).sortBy(-_._2).take(1)(0)._1

    val rocLogReg = metricsLogReg.areaUnderROC()
    println("model ROC = " + rocLogReg.toString)



    println("model ROC = " + rocLogReg.toString)
    println ("positives" + y_positive.toString)
    println ("negatives" + y_negative.toString)
    println ("positives" + (y_positive*1.0/(y_positive+y_negative)).toString)
    println ("negatives" + (y_negative*1.0/(y_positive+y_negative)).toString)


  def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
    val p = new java.io.PrintWriter(f)
    try { op(p) } finally { p.close() }
  }
  import java.util.Calendar
  val today = Calendar.getInstance.getTime


  import java.io._
  val datastr = Array("PCA size: " + PCAsize.toString,
                   "Sample filter: " + sample_filter_val.toString,
                   "Model step: " + step_iter.toString,
                   "",
                   "model ROC = " + rocLogReg.toString,
                   "positives" + y_positive.toString,
                   "negatives" + y_negative.toString,
                   "positives" + (y_positive*1.0/(y_positive+y_negative)).toString,
                   "negatives" + (y_negative*1.0/(y_positive+y_negative)).toString)


  printToFile(new File(dataDir + "example_"+today.toString+".txt")) { p =>
          datastr.foreach(p.println)
        }
        
  
  }
}
