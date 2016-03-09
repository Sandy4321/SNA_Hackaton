cd My_Model_scala/
sbt package
cd ..

spark-1.6.0/bin/spark-submit --class "Data_test" --master local[8] --driver-memory 8G My_Model_scala/target/scala-2.10/baseline_2.10-1.0.jar Data_short/
