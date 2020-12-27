package com.packt.JavaDL.TitanicSurvival.Util;

import static org.apache.spark.sql.functions.col;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.sql.types.DataTypes;

import scala.Option;
import scala.Some;

public class DataSetCreator {
    public static UDF1<String, Option<Integer>> normSex = (String d) -> {
        if (null == d)
            return Option.apply(null);
        else {
          if (d.equals("male"))
              return Some.apply(0);
          else
              return Some.apply(1);
        }
      };

      public static UDF1<String, Option<Integer>> normEmbarked = (String d) -> {
          if (null == d)
              return Option.apply(null);
          else {
              if (d.equals("S"))
                  return Some.apply(0);
              else if (d.equals("C"))
                  return Some.apply(1);
              else
                  return Some.apply(2);
          }
      };

	public static void main(String[] args) {
		
		SparkSession spark = SparkSession
	    	      .builder()
	    	      .master("local[*]")
	    	      .config("spark.sql.warehouse.dir", "C:/Exp/")
	    	      .appName("OneVsRestExample")
	    	      .getOrCreate();
	    	      
		Dataset<Row> df = spark.sqlContext()
                .read()
                .format("com.databricks.spark.csv")
                .option("header", "true") // Use first line of all files as header
                .option("inferSchema", "true") // Automatically infer data types
                .load("data/train.csv");
		df.show();
        
        Dataset<Row> df2 = df.select(col("Survived").cast(DataTypes.IntegerType),
        	                         col("Age").cast(DataTypes.IntegerType),
        	                         col("Fare").cast(DataTypes.DoubleType),
        	                         col("Pclass").cast(DataTypes.DoubleType),
        	                         col("Sex"), col("Name"), col("Embarked"));
        
        // We replace the missing values of the age and fare columns by their mean.
        Map<String, Object> m = new HashMap<String, Object>();
        m.put("Age", 30);
        m.put("Fare", 32.2);
        Dataset<Row> trainingDF1 = df.na().fill(m);       
        
        Dataset<Row> trainingDF2 = trainingDF1.drop("PassengerId", "Name", "Ticket", "Cabin");
        trainingDF2.show();
        
        StringIndexer sexIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("sexIndex").setHandleInvalid("skip");
        StringIndexer embarkedIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("embarkedIndex").setHandleInvalid("skip");
        
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {sexIndexer, embarkedIndexer});

        	    // Fit the pipeline to training documents.
        Dataset<Row> trainingDF3 = pipeline.fit(trainingDF2).transform(trainingDF2).drop("Sex", "Embarked");
        trainingDF3.show();
        
        Dataset<Row> finalDF = trainingDF3.select("Pclass", "Age", "SibSp", "Parch", "Fare", "sexIndex", "embarkedIndex", "Survived");
        finalDF.show();
    
	    Dataset<Row>[] splits = finalDF.randomSplit(new double[] {0.7, 0.3});
	    Dataset<Row> trainingData = splits[0];
	    Dataset<Row> testData = splits[1];
	    
	    trainingData.coalesce(1).write().format("com.databricks.spark.csv").option("header", "false").option("delimiter", ",").save("data/Titanic_Train.csv");
	    testData.coalesce(1).write().format("com.databricks.spark.csv").option("header", "false").option("delimiter", ",").save("data/Titanic_Test.csv");
    }
}
