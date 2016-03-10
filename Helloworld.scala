import org.apache.spark.mllib.linalg.Vectors
import scala.collection._	


object TestBuild {

	def main(args: Array[String]) {


	val xx = "%021d".format(0).takeRight(21).map(_.toString().toInt)
	val xx2 = Vectors.dense(1,2,3)

	println ("dsfsdfsf!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
	println (xx)
	println (xx2)
	println (xx +: xx2.toArray)
		 	//map(_.toString().toInt)).toInt
	

}
}