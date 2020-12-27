package YelpImageClassification.Evaluator;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import YelpImageClassification.Preprocessor.FeatureAndDataAligner;
import YelpImageClassification.Preprocessor.makeND4jDataSets;
import YelpImageClassification.Trainer.NetworkSaver;;

public class ResultFileGenerator {
    public static void writeSubmissionFile(String outcsv, List<Pair<String, List<Double>>> phtoObj, double thresh) throws FileNotFoundException {
        try (PrintWriter writer = new PrintWriter(outcsv)) {
            writer.println("business_ids,labels");
            for (int i = 0; i < phtoObj.size(); i++) {
                Pair<String, List<Double>> kv = phtoObj.get(i);
                StringBuffer sb = new StringBuffer();
                Iterator<Double> iter = kv.getValue().stream().filter(x -> x >= thresh).iterator();
                for (int idx = 0; iter.hasNext(); idx++) {
                    iter.next();
                    if (idx > 0) {
                        sb.append(' ');
                    }
                    sb.append(Integer.toString(idx));
                }
                String line = kv.getKey() + "," + sb.toString();
                writer.println(line);
            }
        }
    }

    @SuppressWarnings({ "rawtypes", "unchecked" })
	public static List<Pair<String, List<Double>>> SubmitObj(FeatureAndDataAligner alignedData,
                                               String modelPath,
                                               String model0,
                                               String model1,
                                               String model2,
                                               String model3,
                                               String model4,
                                               String model5,
                                               String model6,
                                               String model7,
                                               String model8) throws IOException {
        List<String> models = Arrays.asList(model0, model1, model2, model3, model4, model5, model6, model7, model8);
        ArrayList<Map<String, Double>> big = new ArrayList<>();
        for (String m : models) {
            INDArray ds = makeND4jDataSets.makeDataSetTE(alignedData);
            MultiLayerNetwork model = NetworkSaver.loadNN(modelPath + m + ".json", modelPath + m + ".bin");
            INDArray scores = ModelEvaluation.scoreModel(model, ds);
            List<Pair<String, Double>> bizScores = ModelEvaluation.aggImgScores2Business(scores, alignedData);
            Map<String, Double> map = bizScores.stream().collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()));
            big.add(map);
        }

        // transforming the data structure above into a List for each bizID containing a Tuple (bizid, List[Double]) where the Vector[Double] is the
        // the vector of probabilities
        List<Pair<String, List<Double>>> result = new ArrayList<>();
        Iterator<String> iter = alignedData.data().stream().map(e -> e.b).distinct().iterator();
        while (iter.hasNext()) {
            String x = iter.next();
            result.add(new MutablePair(x, big.stream().map(x2 -> x2.get(x)).collect(Collectors.toList())));
        }
        return result;
    }
}
