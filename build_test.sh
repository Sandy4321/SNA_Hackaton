#!/bin/sh


cd My_Model_scala/
sbt package
cd ..
# rm -rf    Data_short/trainSubReversedGraph_bin_map
# rm -rf    Data_short/commonFriendsCountsPartitioned_bin_map

spark-1.6.0/bin/spark-submit --class "Map_bin_test" --master local[8] --driver-memory 8G My_Model_scala/target/scala-2.10/baseline_2.10-1.0.jar Data_short/
