package YelpImageClassification.Classifier;

import static YelpImageClassification.Evaluator.ResultFileGenerator.SubmitObj;
import static YelpImageClassification.Evaluator.ResultFileGenerator.writeSubmissionFile;
import static YelpImageClassification.Preprocessor.CSVImageMetadataReader.readBusinessLabels;
import static YelpImageClassification.Preprocessor.CSVImageMetadataReader.readBusinessToImageLabels;
import static YelpImageClassification.Preprocessor.imageFeatureExtractor.getImageIds;
import static YelpImageClassification.Preprocessor.imageFeatureExtractor.processImages;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import YelpImageClassification.Evaluator.ModelEvaluation;
import YelpImageClassification.Preprocessor.FeatureAndDataAligner;
import YelpImageClassification.Preprocessor.makeND4jDataSets;
import YelpImageClassification.Trainer.CNNEpochs;
import YelpImageClassification.Trainer.NetworkSaver;

public class YelpImageClassifier {
    public static void main(String[] args) throws IOException {
        Map<String, Set<Integer>> labMap = readBusinessLabels("C:/Users/admin-karim/Downloads/Yelp/labels/train.csv");
        
        Map<Integer, String> businessMap = readBusinessToImageLabels("C:/Users/admin-karim/Downloads/Yelp/labels/train_photo_to_biz_ids.csv");
        List<String> businessIds = businessMap.entrySet().stream().map(e -> e.getValue()).distinct().collect(Collectors.toList());
        List<String> imgs = getImageIds("C:/Users/admin-karim/Downloads/Yelp/images/train/", businessMap, businessIds).subList(0, 100); // 20000 images
        System.out.println("Image ID retreival done!");

        Map<Integer, List<Integer>> dataMap = processImages(imgs, 64);
        System.out.println("Image processing done!");

        FeatureAndDataAligner alignedData = new FeatureAndDataAligner(dataMap, businessMap, Optional.of(labMap));
        //System.out.println(alignedData.data());
        System.out.println("Feature extraction done!");

        // Training one model for one class at a time
        CNNEpochs.trainModelEpochs(alignedData, 0, "results/models/model0"); 
        CNNEpochs.trainModelEpochs(alignedData, 1, "results/models/model1");
        CNNEpochs.trainModelEpochs(alignedData, 2, "results/models/model2");
        CNNEpochs.trainModelEpochs(alignedData, 3, "results/models/model3");
        CNNEpochs.trainModelEpochs(alignedData, 4, "results/models/model4");
        CNNEpochs.trainModelEpochs(alignedData, 5, "results/models/model5");
        CNNEpochs.trainModelEpochs(alignedData, 6, "results/models/model6");
        CNNEpochs.trainModelEpochs(alignedData, 7, "results/models/model7");
        CNNEpochs.trainModelEpochs(alignedData, 8, "results/models/model8");

        // processing test data for scoring
        Map<Integer, String> businessMapTE = readBusinessToImageLabels("C:/Users/admin-karim/Downloads/Yelp/labels/test_photo_to_biz.csv");
        List<String> imgsTE = getImageIds("C:/Users/admin-karim/Downloads/Yelp/images/test/", businessMapTE, businessMapTE.values().stream().distinct().collect(Collectors.toList())).subList(0, 100);

        Map<Integer, List<Integer>> dataMapTE = processImages(imgsTE, 64); // make them 64x64
        FeatureAndDataAligner alignedDataTE = new FeatureAndDataAligner(dataMapTE, businessMapTE, Optional.empty());

        // creating csv file to submit to kaggle (scores all models)
        List<Pair<String, List<Double>>> Results = SubmitObj(alignedDataTE, "results/models/", "model0", 
        																	"model1", "model2", "model3", "model4", 
        																	"model5", "model6", "model7", "model8");
        writeSubmissionFile("results/kaggleSubmission/kaggleSubmitFile.csv", Results, 0.50);
        
       // example of how to score just model
        INDArray dsTE = makeND4jDataSets.makeDataSetTE(alignedDataTE);
        MultiLayerNetwork model = NetworkSaver.loadNN("results/models/model0.json", "results/models/model0.bin");
        INDArray predsTE = ModelEvaluation.scoreModel(model, dsTE);
        List<Pair<String, Double>> bizScoreAgg = ModelEvaluation.aggImgScores2Business(predsTE, alignedDataTE);
        System.out.println(bizScoreAgg);
    }
}
