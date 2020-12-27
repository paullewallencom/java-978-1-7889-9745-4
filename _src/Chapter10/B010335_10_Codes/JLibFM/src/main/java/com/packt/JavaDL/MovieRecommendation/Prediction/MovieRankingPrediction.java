package com.packt.JavaDL.MovieRecommendation.Prediction;

import static org.ranksys.formats.parsing.Parsers.lp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.function.Function;
import java.util.function.IntPredicate;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.jooq.lambda.Unchecked;
import org.ranksys.fm.PreferenceFM;
import org.ranksys.fm.data.OneClassPreferenceFMData;
import org.ranksys.fm.rec.FMRecommender;
import org.ranksys.formats.index.ItemsReader;
import org.ranksys.formats.index.UsersReader;
import org.ranksys.formats.preference.SimpleRatingPreferencesReader;
import org.ranksys.formats.rec.RecommendationFormat;
import org.ranksys.formats.rec.SimpleRecommendationFormat;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.data.FMData;
import org.ranksys.javafm.learner.gd.PointWiseError;

import com.packt.JavaDL.MovieRecommendation.FMCore.PointWiseGradientDescent;

import es.uam.eps.ir.ranksys.fast.index.FastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.FastUserIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastItemIndex;
import es.uam.eps.ir.ranksys.fast.index.SimpleFastUserIndex;
import es.uam.eps.ir.ranksys.fast.preference.FastPreferenceData;
import es.uam.eps.ir.ranksys.fast.preference.SimpleFastPreferenceData;
import es.uam.eps.ir.ranksys.rec.Recommender;
import es.uam.eps.ir.ranksys.rec.runner.RecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilterRecommenderRunner;
import es.uam.eps.ir.ranksys.rec.runner.fast.FastFilters;

/*
 * @author Md. Rezaul Karim, 07/06/2018
 */

public class MovieRankingPrediction {
    public static void main(String... args) throws IOException {
        final String folderPath = "ml-1m";
        final String indexPath = "index";
        final String userFileName = "users.dat";
        final String moviesFileName = "movies.dat";
        final String ratingsFileName = "ratings.dat";
        final String encodingUTF8 = "UTF-8";

        final String userDatPath = folderPath + "/" + userFileName;
        final String movieDatPath = folderPath + "/" + moviesFileName;

        final String userIndexPath = indexPath + "/" + "userIndex";
        final String movieIndexPath = indexPath + "/" + "movieIndex";

        if ( !Files.exists(Paths.get(userIndexPath))) {
            createIndexFromFile(userDatPath, encodingUTF8, userIndexPath);
        }

        if ( !Files.exists(Paths.get(movieIndexPath))) {
            createIndexFromFile(movieDatPath, encodingUTF8, movieIndexPath);
        }

        String trainDataPath = indexPath + "/ratings_train";
        String testDataPath = indexPath + "/ratings_test";
        final String ratingsDatPath = folderPath + "/" + ratingsFileName;
        
        if ( !Files.exists(Paths.get(trainDataPath))) {
            generateTrainAndTestDataSet(ratingsDatPath, trainDataPath, testDataPath);
        }

        // This creates the index for a set of users. Here the users are internally represented with numerical indices from 0 (inclusive) to the number of indexed users (exclusive).
        FastUserIndex<Long> userIndex = SimpleFastUserIndex.load(UsersReader.read(userIndexPath, lp));
        
        // This creates the index for a set of items. Here the items are internally represented with numerical indices from 0 (inclusive) to the number of indexed items (exclusive).
        FastItemIndex<Long> itemIndex = SimpleFastItemIndex.load(ItemsReader.read(movieIndexPath, lp));

        // Store the preferences/rating for users and items provided by FastUserIndex and FastItemIndex.
        FastPreferenceData<Long, Long> trainData = SimpleFastPreferenceData.load(SimpleRatingPreferencesReader.get().read(trainDataPath, lp, lp), userIndex, itemIndex);
        FastPreferenceData<Long, Long> testData = SimpleFastPreferenceData.load(SimpleRatingPreferencesReader.get().read(testDataPath, lp, lp), userIndex, itemIndex);

        // Create a Recommender interface that will be used by FMRecommender.
        Map<String, Supplier<Recommender<Long, Long>>> recMap = new HashMap<>();

        // Use Factorisation machine that uses RMSE-like loss with balanced sampling of negative instances:
        String outFileName = "outFolder/Ranking_RMSE.txt";
        recMap.put(outFileName, Unchecked.supplier(() -> {
            double negativeProp = 2.0D;
            
            FMData fmTrain = new OneClassPreferenceFMData(trainData, negativeProp);
            FMData fmTest = new OneClassPreferenceFMData(testData, negativeProp);
            
            double learnRate = 0.01D; // Learning Rate
            int numIter = 10; // Number of Iterations
            double sdev = 0.1D;
            double regB = 0.01D;
            
            double[] regW = new double[fmTrain.numFeatures()];
            Arrays.fill(regW, 0.01D);
            double[] regM = new double[fmTrain.numFeatures()];
            
            Arrays.fill(regM, 0.01D);
            int K = 100;
            
            FM fm = new FM(fmTrain.numFeatures(), K, new Random(), sdev);
            (new PointWiseGradientDescent(learnRate, numIter, PointWiseError.rmse(), regB, regW, regM)).learn(fm, fmTrain, fmTest);
            PreferenceFM<Long, Long> prefFm = new PreferenceFM<Long, Long>(userIndex, itemIndex, fm);
            
            return new FMRecommender<Long, Long>(prefFm);
        }));
        
        //System.out.print("Prediction has been saved at " + outFileName);
        Set<Long> targetUsers = testData.getUsersWithPreferences().collect(Collectors.toSet());

        // SimpleRecommendationFormat is used here for the recommendations format, which is in tab-separated user-item-score triplets: 
        RecommendationFormat<Long, Long> format = new SimpleRecommendationFormat<>(lp, lp);
        Function<Long, IntPredicate> filter = FastFilters.notInTrain(trainData);
        int maxLength = 100;

        // Generate recommendations and print it based on the format.
        RecommenderRunner<Long, Long> runner = new FastFilterRecommenderRunner<>(userIndex, itemIndex, targetUsers.stream(), filter, maxLength);
        
        recMap.forEach(Unchecked.biConsumer((name, recommender) -> {
            System.out.println("Ranking prediction is ongoing...");
            System.out.println("Result will be saved at " + name);
            try (RecommendationFormat.Writer<Long, Long> writer = format.getWriter(name)) {
                runner.run(recommender.get(), writer);
            }
        }));
        
        System.out.println("Graph plotting ...");
        System.out.println("Prediction has been saved at " + outFileName);
    }
    
    static void createIndexFromFile(String fileReadPath, String encodings, String fileWritePath) throws IOException {
        BufferedReader bufferedReader = new BufferedReader(
                new InputStreamReader(new FileInputStream(
                        fileReadPath), Charset.forName(encodings)));

        BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(
                        new FileOutputStream(fileWritePath)));

        String line;
        while ((line = bufferedReader.readLine()) != null) {
            StringBuilder builder = new StringBuilder();
            String[] lineArray = line.split("::");
            builder.append(lineArray[0]);
            writer.write(builder.toString());
            writer.newLine();
        }
        
        writer.flush();

        bufferedReader.close();
        writer.close();
    }

    static void generateTrainAndTestDataSet(String ratingsDatPath, String trainDataPath, String testDataPath) throws IOException {
        BufferedWriter writerTrain = new BufferedWriter(
                new OutputStreamWriter(
                        new FileOutputStream(trainDataPath)));

        BufferedWriter writerTest = new BufferedWriter(
                new OutputStreamWriter(
                        new FileOutputStream(testDataPath)));

        BufferedReader bufferedReader = new BufferedReader(new FileReader(ratingsDatPath));
        List<String> dummyData = new ArrayList<>();
        String line;
        while ((line = bufferedReader.readLine()) != null) {
            String removeDots = line.replaceAll("::", "\t");
            dummyData.add(removeDots);
        }
        
        bufferedReader.close();

        Random generator = new Random();
        int dataSize = dummyData.size();
        int trainDataSize = (int)(dataSize * (2.0 / 3.0));
        int i = 0;
        
        while(i < trainDataSize){
            int random = generator.nextInt(dummyData.size()-0) + 0;
            line = dummyData.get(random);
            dummyData.remove(random);
            writerTrain.write(line);
            writerTrain.newLine();
            i++;
        }

        int j = 0;
        while(j < (dataSize - trainDataSize)){
            writerTest.write(dummyData.get(j));
            writerTest.newLine();
            j++;
        }

        writerTrain.flush();
        writerTrain.close();

        writerTest.flush();
        writerTest.close();
    }
}
