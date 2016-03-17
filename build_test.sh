#!/bin/sh


cd My_Model_scala/
sbt package
cd ..


rm -rf Data_short/trainSubReversedGraph/
rm -rf Data_short/LogisticRegressionModel/
rm -rf Data_short/prediction/
rm -rf Data_short/commonFriendsCountsPartitioned/
rm -rf Data_short/trainSubReversedGraph_bin_map/
rm -rf Data_short/commonFriendsCountsPartitioned_bin_map


spark-1.6.0/bin/spark-submit --class "Baseline_full_v1" --master local[*] --driver-memory 24G My_Model_scala/target/scala-2.10/baseline_2.10-1.0.jar Data_short/
