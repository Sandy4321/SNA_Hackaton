#!/bin/sh


cd My_Model_scala/
sbt package
cd ..
#rm -rf Data/trainSubReversedGraph/
rm -rf Data/LogisticRegressionModel/
rm -rf Data/prediction/

rm -rf Data_short/LogisticRegressionModel/
rm -rf Data_short/prediction/
rm -rf Data_short/validPrediction
rm -rf Data_short/validPrediction_real

#rm -rf Data/commonFriendsCountsPartitioned/
#rm -rf Data/trainSubReversedGraph_bin_map/
#rm -rf Data_short/commonFriendsCountsPartitioned_bin_map


spark-1.6.0/bin/spark-submit --class "Baseline_full_v1_testing_iter" --master local[*] --driver-memory 24G My_Model_scala/target/scala-2.10/baseline_2.10-1.0.jar Data_short/
