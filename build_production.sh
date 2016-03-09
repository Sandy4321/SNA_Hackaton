cd My_Model_scala/
sbt package
cd ..
rm -rf Data_short/trainSubReversedGraph/
rm -rf Data_short/LogisticRegressionModel/
rm -rf Data_short/prediction/
rm -rf Data_short/commonFriendsCountsPartitioned/

spark-1.6.0/bin/spark-submit --class "Baseline" --master local[8] --driver-memory 8G My_Model_scala/target/scala-2.10/baseline_2.10-1.0.jar Data/
