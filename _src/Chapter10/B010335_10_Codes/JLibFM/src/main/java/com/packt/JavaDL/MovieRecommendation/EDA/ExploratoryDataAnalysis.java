package com.packt.JavaDL.MovieRecommendation.EDA;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.SparkSession.Builder;

/*
 * @author Md. Rezaul Karim, 07/06/2018
 */

public class ExploratoryDataAnalysis {
    public static void main(String... args) {
        String ratingsFile = "ml-small/ratings.csv";
        String movieFile  = "ml-small/movies.csv";
        
        SparkSession spark = new Builder()
        	      .master("local[*]")
        	      .config("spark.sql.warehouse.dir", "temp/")
        	      .appName("Bitcoin Preprocessing")
        	      .getOrCreate();

        // Read RatingsFile
        Dataset<Row> df1 = spark.read()
                .format("com.databricks.spark.csv")
                .option("inferSchemea", "true")
                .option("header", "true")
                .load(ratingsFile);

        Dataset<Row> ratingsDF = df1.select(df1.col("userId"), df1.col("movieId"),
                df1.col("rating"), df1.col("timestamp"));
        ratingsDF.show(10);

        // Read MoviesFile
        Dataset<Row> df2 = spark.read()
                .format("com.databricks.spark.csv")
                .option("inferSchema", "true")
                .option("header", "true")
                .load(movieFile);
        
        Dataset<Row> moviesDF = df2.select(df2.col("movieId"), df2.col("title"),
                df2.col("genres"));
        moviesDF.show(10);

        ratingsDF.createOrReplaceTempView("ratings");
        moviesDF.createOrReplaceTempView("movies");

        long numberOfRatings = ratingsDF.count();
        long numberOfUsers   = ratingsDF.select(ratingsDF.col("userId")).distinct().count();
        long numberOfMovies  = ratingsDF.select(ratingsDF.col("movieId")).distinct().count();
        String print = String.format("Got %d ratings from %d users on %d movies.", numberOfRatings, numberOfUsers, numberOfMovies);
        
        // "Got 100004 ratings from 671 users on 9066 movies"
        System.out.println(print);

        // Get the max, min ratings along with the count of users who have rated a movie.
        Dataset<Row> sqlDF = spark.sql(
                "SELECT movies.title, movierates.maxr, movierates.minr, movierates.cntu "
                        + "FROM (SELECT "
                        + "ratings.movieId, MAX(ratings.rating) AS maxr,"
                        + "MIN(ratings.rating) AS minr, COUNT(distinct userId) AS cntu "
                        + "FROM ratings "
                        + "GROUP BY ratings.movieId) movierates "
                        + "JOIN movies ON movierates.movieId=movies.movieId "
                        + "ORDER BY movierates.cntu DESC");
        sqlDF.show(10);
        /*
        +------------------------------------------------------------------------------+----+----+----+
        |title                                                                         |maxr|minr|cntu|
        +------------------------------------------------------------------------------+----+----+----+
        |Forrest Gump (1994)                                                           |5.0 |1.0 |341 |
        |Pulp Fiction (1994)                                                           |5.0 |0.5 |324 |
        |Shawshank Redemption, The (1994)                                              |5.0 |1.0 |311 |
        |Silence of the Lambs, The (1991)                                              |5.0 |0.5 |304 |
        |Star Wars: Episode IV - A New Hope (1977)                                     |5.0 |0.5 |291 |
        |Jurassic Park (1993)                                                          |5.0 |0.5 |274 |
        |Matrix, The (1999)                                                            |5.0 |1.0 |259 |
        |Toy Story (1995)                                                              |5.0 |1.0 |247 |
        |Schindler's List (1993)                                                       |5.0 |0.5 |244 |
        |Terminator 2: Judgment Day (1991)                                             |5.0 |1.0 |237 |
        |Star Wars: Episode V - The Empire Strikes Back (1980)                         |5.0 |0.5 |234 |
        |Braveheart (1995)                                                             |5.0 |0.5 |228 |
        |Back to the Future (1985)                                                     |5.0 |1.0 |226 |
        |Fargo (1996)                                                                  |5.0 |1.0 |224 |
        |Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)|5.0 |0.5 |220 |
        |American Beauty (1999)                                                        |5.0 |0.5 |220 |
        |Independence Day (a.k.a. ID4) (1996)                                          |5.0 |0.5 |218 |
        |Star Wars: Episode VI - Return of the Jedi (1983)                             |5.0 |0.5 |217 |
        |Aladdin (1992)                                                                |5.0 |0.5 |215 |
        |Fugitive, The (1993)                                                          |5.0 |0.5 |213 |
        +------------------------------------------------------------------------------+----+----+----+
        only showing top 20 rows
        */

        // Top 10 active users and how many times they rated a movie.
        Dataset<Row> mostActiveUsersSchemaRDD = spark.sql(
                "SELECT ratings.userId, count(*) AS ct "
                        + "FROM ratings "
                        + "GROUP BY ratings.userId "
                        + "ORDER BY ct DESC LIMIT 10");
        mostActiveUsersSchemaRDD.show(10);
        
        /*
        +------+----+
        |userId|ct  |
        +------+----+
        |547   |2391|
        |564   |1868|
        |624   |1735|
        |15    |1700|
        |73    |1610|
        |452   |1340|
        |468   |1291|
        |380   |1063|
        |311   |1019|
        |30    |1011|
        +------+----+
        */

        // Movies that user 668 rated higher than 4
        Dataset<Row> userRating = spark.sql(
                "SELECT ratings.userId, ratings.movieId, ratings.rating, movies.title "
                        + "FROM ratings JOIN movies "
                        + "ON movies.movieId=ratings.movieId "
                        + "WHERE ratings.userId=668 AND ratings.rating > 4");
        userRating.show(10);
        
        /*
        +------+-------+------+------------------------------------------+
        |userId|movieId|rating|title                                     |
        +------+-------+------+------------------------------------------+
        |668   |296    |5.0   |Pulp Fiction (1994)                       |
        |668   |593    |5.0   |Silence of the Lambs, The (1991)          |
        |668   |608    |5.0   |Fargo (1996)                              |
        |668   |1213   |5.0   |Goodfellas (1990)                         |
        |668   |1221   |5.0   |Godfather: Part II, The (1974)            |
        |668   |2324   |5.0   |Life Is Beautiful (La Vita è bella) (1997)|
        |668   |2908   |5.0   |Boys Don't Cry (1999)                     |
        |668   |2997   |5.0   |Being John Malkovich (1999)               |
        +------+-------+------+------------------------------------------+
        */
    }
}
