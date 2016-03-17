#!/bin/sh


cd My_Model_scala/
sbt package
cd ..
rm -rf Data/trainSubReversedGraph/
rm -rf Data/LogisticRegressionModel/
rm -rf Data/prediction/
rm -rf Data/commonFriendsCountsPartitioned/
rm -rf Data/trainSubReversedGraph_bin_map/
rm -rf Data_short/commonFriendsCountsPartitioned_bin_map


spark-1.6.0/bin/spark-submit --class "Baseline_full_v1" --master local[8] --driver-memory 24G My_Model_scala/target/scala-2.10/baseline_2.10-1.0.jar Data/
