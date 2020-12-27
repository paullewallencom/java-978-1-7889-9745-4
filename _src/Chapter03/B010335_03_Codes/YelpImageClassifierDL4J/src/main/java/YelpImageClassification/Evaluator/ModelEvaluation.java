package YelpImageClassification.Evaluator;

import YelpImageClassification.Preprocessor.FeatureAndDataAligner;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class ModelEvaluation {
    public static INDArray scoreModel(MultiLayerNetwork model, INDArray ds) {
        return model.output(ds);
    }

    /** Take model predictions from scoreModel and merge with alignedData*/
    public static List<Pair<String, Double>> aggImgScores2Business(INDArray scores,
                                                                   FeatureAndDataAligner alignedData) {
        assert(scores.size(0) == alignedData.data().size());
        ArrayList<Pair<String, Double>> result = new ArrayList<Pair<String, Double>>();
        for (String x : alignedData.getBusinessIds().stream().distinct().collect(Collectors.toList())) {
            //R irows = getRowIndices4Business(alignedData.getBusinessIds(), x);
            List<String> ids = alignedData.getBusinessIds();
            DoubleStream ret = IntStream.range(0, ids.size())
                    .filter(i -> ids.get(i).equals(x))
                    .mapToDouble(e -> scores.getRow(e).getColumn(1).getDouble(0,0));
            double mean = ret.sum() / ids.size();
            result.add(new ImmutablePair<>(x, mean));

        }
        return result;
    }
}
