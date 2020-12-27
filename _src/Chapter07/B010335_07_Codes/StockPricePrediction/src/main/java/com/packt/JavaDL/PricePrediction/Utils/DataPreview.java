package com.packt.JavaDL.PricePrediction.Utils;

import java.io.IOException;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class DataPreview {
    public static void main (String[] args) throws IOException {
        SparkSession spark = SparkSession.builder().master("local").appName("DataProcess").getOrCreate();
        spark.conf().set("spark.sql.crossJoin.enabled", "true");
        String filename = "data/prices-split-adjusted.csv";

        // load data from csv file
        Dataset<Row> data = spark.read().option("inferSchema", false).option("header", true).format("csv").load(filename)
                .withColumn("openPrice", functions.col("open").cast("double")).drop("open")
                .withColumn("closePrice", functions.col("close").cast("double")).drop("close")
                .withColumn("lowPrice", functions.col("low").cast("double")).drop("low")
                .withColumn("highPrice", functions.col("high").cast("double")).drop("high")
                .withColumn("volumeTmp", functions.col("volume").cast("double")).drop("volume")
                .toDF("date", "symbol", "open", "close", "low", "high", "volume");

        data.show(10);
        
        data.createOrReplaceTempView("stock");
        
        spark.sql("SELECT DISTINCT symbol FROM stock GROUP BY symbol").show(10); 
        
        spark.sql("SELECT symbol, avg(open) as avg_open, "
        		+ "avg(close) as avg_close, "
        		+ "avg(low) as avg_low, "
        		+ "avg(high) as avg_high "
        		+ "FROM stock GROUP BY symbol")
        		.show(10); 
        
        spark.sql("SELECT symbol, "
        		+ "MIN(open) as min_open, MAX(open) as max_open, "
        		+ "MIN(close) as min_close, MAX(close) as max_close, "
        		+ "MIN(low) as min_low, MAX(low) as max_low, "
        		+ "MIN(high) as min_high, MAX(high) as max_high "
        		+ "FROM stock GROUP BY symbol")
        		.show(10);         

        long count = data.select("symbol").count();
        System.out.println("Number of Symbols: " + count);
    }
}
