/**
  * Baseline for hackaton


  5. build project sbt package
  ... 
  ...
  9. send jar you made in step 5 to spark (configuration is given for 4 cores)
  ...
   spark-1.6.0/bin/spark-submit --class "Baseline" --master local[8] --driver-memory 8G My_Model_scala/target/scala-2.10/baseline_2.10-1.0.jar Data_short/


   */


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

import scala.collection.mutable.ArrayBuffer

case class PairWithCommonFriends(person1: Int, person2: Int, commonFriendsCount: Int)
case class UserFriends(user: Int, friends: Array[Int])
case class AgeSex(age: Int, sex: Int)
case class UserCity(city: Int, city_active: Int)
case class RocVals(reg_par: Double, roc_val: Double)
case class Friend(user: Int, mask_bit: Int)
case class UserFriends2(user: Int, friends: Array[Friend])

object Baseline {

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
    val modelPath = dataDir + "LogisticRegressionModel"
    val numPartitions = 200
    val numPartitionsGraph = 107

    //  val numPartitionsGraph = 10

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

    // prepare data for training model
    // step 2




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
                     ageSexBC:  Broadcast[scala.collection.Map[Int, AgeSex ]],
                     cityRegBC: Broadcast[scala.collection.Map[Int, UserCity]],
                     friendscountBC: Broadcast[scala.collection.Map[Int, Int]]) = {

      val zero_fr_masks_lst = "%021d".format(0).takeRight(21).map(_.toString().toInt)
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
        .leftOuterJoin(friend_masks)
        .map(x => x._1 -> (x._2._1.toArray.deep.union(x._2._2.getOrElse(zero_fr_masks_lst))))  // join with friend_masks
        .map(x => x._1 -> (Vectors.dense(x._2.toArray.map({l => l.toString().toDouble}))))  // convert back to vector_dense
        .leftOuterJoin(positives)
        
    }


    //
    // if point class is not positive than we make it zero
    //
    val data = {
      prepareData(commonFriendsCounts, positives, friend_masks, ageSexBC, cityRegBC, friendscountBC)
        .map(t => LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
    }


    // println ("\n\n\n\n\n")
    // println ("\n\n\n\n\n")
    // data.take(20).map(t => println(t))
    // println ("\n\n\n\n\n")
    // println ("\n\n\n\n\n")
    // println ("\n\n\n\n\n")



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

    // https://www.kaggle.com/rootua/avito-context-ad-clicks/apache-spark-scala-logistic-regression/run/27034
    val model_not_trained =  new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true)
    model_not_trained.optimizer.setNumIterations(3000)
    val model = model_not_trained.run(training)


    model.clearThreshold()
    model.save(sc, modelPath)

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

    // compute scores on the test set
    // step 7
    val testCommonFriendsCounts = {
      sqlc
        .read.parquet(commonFriendsPath + "/part_*/")
        .map(t => PairWithCommonFriends(t.getAs[Int](0), t.getAs[Int](1), t.getAs[Int](2)))
        .filter(pair => pair.person1 % 11 == 7 || pair.person2 % 11 == 7)
    }

    val testData = {
      prepareData(testCommonFriendsCounts, positives, friend_masks, ageSexBC,cityRegBC,friendscountBC)
        .map(t => t._1 -> LabeledPoint(t._2._2.getOrElse(0.0), t._2._1))
        .filter(t => t._2.label == 0.0)
        .map(t => t._1 -> LabeledPoint(t._2.label, scaler1.transform(t._2.features)))
    }

    // step 8
    val testPrediction = {
      testData
        .flatMap { case (id, LabeledPoint(label, features)) =>
          val prediction = model.predict(features)
          Seq(id._1 -> (id._2, prediction), id._2 -> (id._1, prediction))
        }
        .filter(t => t._1 % 11 == 7 && t._2._2 >= threshold)
        .groupByKey(numPartitions)
        .map(t => {
          val user = t._1
          val firendsWithRatings = t._2
          val topBestFriends = firendsWithRatings.toList.sortBy(-_._2).take(100).map(x => x._1)
          (user, topBestFriends)
        })
        .sortByKey(true, 1)
        .map(t => t._1 + "\t" + t._2.mkString("\t"))
    }

    testPrediction.saveAsTextFile(predictionPath,  classOf[GzipCodec])

    println("model ROC = " + rocLogReg.toString)
    println ("positives" + y_positive.toString)
    println ("negatives" + y_negative.toString)
    println ("positives" + (y_positive*1.0/(y_positive+y_negative)).toString)
    println ("negatives" + (y_negative*1.0/(y_positive+y_negative)).toString)

  }
}
