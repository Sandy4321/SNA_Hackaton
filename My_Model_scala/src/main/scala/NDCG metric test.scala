
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
    val numPartitions = 200
    //val numPartitionsGraph = 107
    //val numPartitionsGraph = 20


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


    //val graph = {
    
    val graph = {sc.textFile(validationPredictionPath)
                   .map(line => {
                    // val userId = line.nonEmpty
                    var userId: Int =  0
                    var predicted: Array[Int] = Array(0,0,0)

                    if (line.nonEmpty){
                        val ids = line.split("""\s+""")
                        userId = ids.head.toInt
                        predicted = ids.drop(1).map({l => l.toInt})    
                        }
                    
                    (userId, predicted)
                    }

                ).filter(t => t._1!=0)
               }
    

    //graph.take(50).map(t => println (t))

    val graph_real_par1 = {sc.textFile(validationPredictionPath+"_real")
                    .map(line => {
                        val lineSplit = line.replace("(", "").replace(")", "").split(",")
                        (lineSplit(0).toInt,lineSplit(1).toInt)

                        })}
    val graph_real = graph_real_par1.map(t => t._1 -> t._2).union(graph_real_par1.map(t => t._2 -> t._1)).groupByKey()

    graph_real.take(20).map(println)

    println(graph.count)
    println(graph_real.count)
    println (Iterator(1,0,1,1).map(t => t==1).map(if (_) 1d else 0d).toArray.toVector.map(_.toString))


    val evaluate_score: Double = {graph


        
    //

    // def evaluate(file: File): Double = {
    //   val inputStream = new FileInputStream(file)
    //   try {
    //     val src = Source.fromInputStream(new GZIPInputStream(inputStream)).getLines()
    //     var accum: Double = 0d
    //     for (line <- src.map(_.trim) if line.nonEmpty) {
    //       val ids = line.split("""\s+""")
    //       val userId: Int = ids.head.toInt
    //       val predicted: Iterator[Int] = ids.iterator.drop(1).map(_.toInt)
    //       for (actual <- evaluationSet.get(userId)) {     // Ignore users having no hidden links
    //         val real = predicted.map(actual.contains).map(if (_) 1d else 0d)
    //         val rdcg = dcg(real)
    //         val ideal = (0 until actual.size).iterator.map(_ => 1d) // Ideal output contains all hidden links of this user
    //         val idcg = dcg(ideal)
    //         accum += rdcg / idcg
    //       }
    //     }
    //     accum / evaluationSet.size * 1000       // Divide by # of users having hidden links
    //   } finally {
    //     inputStream.close()
    //   }
    // }






            
    }

}