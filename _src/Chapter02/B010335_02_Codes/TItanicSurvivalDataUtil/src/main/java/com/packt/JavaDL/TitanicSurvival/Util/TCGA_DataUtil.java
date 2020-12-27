package com.packt.JavaDL.TitanicSurvival.Util;

import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import static org.apache.spark.sql.functions.col;

public class TCGA_DataUtil {
public static void main(String[] args) {		
		SparkSession spark = SparkSession
	    	      .builder()
	    	      .master("local[*]")
	    	      .config("spark.sql.warehouse.dir", "C:/Exp/")
	    	      .appName("OneVsRestExample")
	    	      .getOrCreate();
		spark.conf().set("spark.sql.crossJoin.enabled", "true");
	    	      
		Dataset<Row> data = spark.read()
                .option("maxColumns", 25000)
                .format("com.databricks.spark.csv")
                .option("header", "true") // Use first line of all files as header
                .option("inferSchema", "true") // Automatically infer data types
                .load("C:/Users/admin-karim/Desktop/TCGA-PANCAN/TCGA-PANCAN-HiSeq-801x20531/data.csv");
		data.show();
		
		int numFeatures = data.columns().length;
		long numSamples = data.count();
		
		System.out.println("Number of features: " + numFeatures);
		System.out.println("Number of samples: " + numSamples);		

		Dataset<Row> numericDF = data.drop("id");// now we have only 20531 features left 	
		
		Dataset<Row> labels = spark.read()
                .format("com.databricks.spark.csv")
                .option("header", "true") // Use first line of all files as header
                .option("inferSchema", "true") // Automatically infer data types
                .load("C:/Users/admin-karim/Desktop/TCGA-PANCAN/TCGA-PANCAN-HiSeq-801x20531/labels.csv");
		labels.show(10);
		
		StringIndexer indexer = new StringIndexer()
								.setInputCol("Class")
								.setOutputCol("label")
								.setHandleInvalid("skip");
		
		Dataset<Row> indexedDF = indexer.fit(labels).transform(labels).select(col("label").cast(DataTypes.IntegerType));
		indexedDF.show(10);
		
		Dataset<Row> combinedDF = numericDF.join(indexedDF);		
	    
	    Dataset<Row>[] splits = combinedDF.randomSplit(new double[] {0.7, 0.3});
	    Dataset<Row> trainingData = splits[0];
	    System.out.println(trainingData.count());
	    
	    Dataset<Row> testData = splits[1];
	    System.out.println(testData.count());
	    
	    //trainingData.coalesce(1).write().format("com.databricks.spark.csv").option("header", "false").option("delimiter", ",").save("data/TCGA_train.csv");
	    //testData.coalesce(1).write().format("com.databricks.spark.csv").option("header", "false").option("delimiter", ",").save("data/TCGA_test.csv");
	    
}

}
