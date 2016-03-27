
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



case class UserPredictionsProba(userId: Int, probability: Double)
case class UserPredictions(userId: Int, predicted: Array[Int])


object NDCG_test {

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
    val validationPredictionPath = dataDir + "validPrediction"
    val NDCGPredictionPath = dataDir + "NDCGPrediction"
    val numPartitions = 200


  import java.io._

    val logOf2 = math.log(2)
    def log2(d: Double): Double = math.log(d) / logOf2
    def dcg(values: Iterator[Double]): Double = {
              var res: Double = 0
              for ((v, i) <- values.zipWithIndex) {
                val gain = (math.pow(2, v) - 1) / log2(i + 2)
                res += gain
              }
              res
        }

    
    // Reading probabilities of having particular edge
    //


   val graph_proba = {sqlc.read.parquet(validationPredictionPath +"_proba")
                            .map(t => t.getAs[Int](0) -> (t.getAs[Int](1), t.getAs[Double](2)))
                       }



   val graph11_proba = {sqlc.read.parquet(predictionPath +"_proba")
                            .map(t => t.getAs[Int](0) -> (t.getAs[Int](1), t.getAs[Double](2)))
                       }




    // Reading resulting predictions
    //
    val threshold = 0.0
    val total_count = 100

    val graph_predictions = {       
            graph_proba
            .filter(t => t._2._2 >= threshold) 
            .groupByKey(numPartitions)
            .map(t => {
              val user = t._1
              val friendsWithRatings = t._2
              val topBestFriends = friendsWithRatings.toList.sortBy(-_._2).take(total_count).map(x => x._1)
              (user, topBestFriends.toArray)
            })
            .sortByKey(true, 1)
            //.map(t => t._1 -> t._2.toVector) 
            .map(t => UserPredictions(t._1, t._2))
            
            }




    val graph11_predictions = {       
            graph11_proba
            .filter(t =>  t._1 % 11 == 7 && t._2._2 >= threshold) 
            .groupByKey(numPartitions)
            .map(t => {
              val user = t._1
              val friendsWithRatings = t._2
              val topBestFriends = friendsWithRatings.toList.sortBy(-_._2).take(total_count).map(x => x._1)
              (user, topBestFriends.toArray)
            })
            .sortByKey(true, 1)
            //.map(t => t._1 -> t._2.toVector) 
            .map(t => UserPredictions(t._1, t._2))
            
            }


    graph11_predictions.map(t => t.userId.toString + "\t" + t.predicted.mkString("\t")).saveAsTextFile(NDCGPredictionPath,  classOf[GzipCodec])




    val graph = {sc.textFile(validationPredictionPath)
                   .map(line => {
                    var userId: Int =  0
                    var predicted: Array[Int] = Array(0,0,0)

                    if (line.nonEmpty){
                        val ids = line.split("""\s+""")
                        userId = ids.head.toInt
                        predicted = ids.drop(1).map({l => l.toInt})    
                        }
                    
                    UserPredictions(userId, predicted)
                    }

                ).filter(t => t.userId!=0)
                   //.map (t => t.userId -> t.predicted.toVector)
                   //.sortByKey()
                   //.map (t => t._2)
               }
    




    // Collecting real connections for graph
    val graph_real_par1 = {sc.textFile(validationPredictionPath+"_real")
                    .map(line => {
                        val lineSplit = line.replace("(", "").replace(")", "").split(",")
                        (lineSplit(0).toInt,lineSplit(1).toInt)
                        
                        })}



    // Graph_real structure:
    // --- (12672400,Vector(50849344, 47234864, 36199872))
    // 
    val graph_real = {graph_real_par1.map(t => t._1 -> t._2)
                        .union(graph_real_par1
                                              .map(t => t._2 -> t._1))
                        .groupByKey()
                        .map(t => t._1.toInt -> t._2.toVector)
                        .sortByKey()
                        .collectAsMap()
                        }



    // ---
    // --- Evaluating score using NDCG metric
    // ---

    val evaluate_score = {
          graph_predictions
        //ignore users having no hidden links
          .filter (t => graph_real.getOrElse(t.userId, Vector(0)) !=0)
          .map (t => {
                    val eval_set: Vector[Int] = graph_real.getOrElse(t.userId, Vector(0))
                    val real = t.predicted.map(eval_set.contains).map(if (_) 1d else 0d)
                    val rdcg = dcg(real.iterator)
                    val ideal = (0 until eval_set.size).iterator.map(_ => 1d) // Ideal output contains all hidden links of this user
                    val idcg = dcg(ideal)
                    val accum = rdcg / idcg

                    t.userId -> accum
                    } )
             .filter(t => !t._2.isNaN)
    }


    val count_greal = graph_real.size
    println ("Evaluation score: ", evaluate_score.map(t => t._2).reduce((a, b) => a + b)/count_greal *1000)






            
    }

}