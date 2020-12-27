package YelpImageClassification.Trainer;

import static YelpImageClassification.Preprocessor.makeND4jDataSets.makeDataSet;

import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import YelpImageClassification.Preprocessor.FeatureAndDataAligner;

public class CNNEpochs {
	public static void trainModelEpochs(FeatureAndDataAligner alignedData, int businessClass, String saveNN) throws IOException {
		DataSet ds = makeDataSet(alignedData, businessClass);

		Logger log = LoggerFactory.getLogger(CNNEpochs.class);
		// log.info("Begin time: " + java.util.Calendar.getInstance().getTime());

		double nfeatures = ds.getFeatures().getRow(0).length(); // hyper, hyper parameter
		// System.out.println(nfeatures);

		int nlabels = ds.getLabels().getRow(0).length();
		System.out.println(nlabels);

		int numRows = (int) Math.sqrt(nfeatures); // numRows * numColumns must equal columns in initial data * channels
		int numColumns = (int) Math.sqrt(nfeatures); // numRows * numColumns must equal columns in initial data *
														// channels

		System.out.println(numRows);
		System.out.println(numColumns);

		int nChannels = 1; // would be 3 if color image w R,G,B
		int outputNum = 2;// # of classes (# of columns in output)
		int seed = 12345;
		int listenerFreq = 1;
		int nepochs = 1000;
		int nbatch = 128; // recommended between 16 and 128

		// nOutPar = 500 // default was 1000. # of output nodes in first layer
		// System.out.println("rows: " + ds.getFeatures().size(0));
		// System.out.println("columns: " + ds.getFeatures().size(1));

		/**
		 * Set a neural network configuration with multiple layers
		 */
		// log.info("Load data....");
		ds.normalize(); // A data transform (example/outcome pairs). The outcomes are specifically for
						// neural network encoding such that any labels that are considered true are 1s.
						// The rest are zeros.
		// System.out.println("Loaded " + ds.labelCounts());
		Nd4j.shuffle(ds.getFeatureMatrix(), new Random(seed), 1); // this shuffles rows in the ds.
		Nd4j.shuffle(ds.getLabels(), new Random(seed), 1); // this shuffles the labels accordingly
		SplitTestAndTrain trainTest = ds.splitTestAndTrain(75, new Random(seed)); // Random Seed not needed here

		// creating epoch dataset iterator
		ListDataSetIterator<DataSet> dsiterTr = new ListDataSetIterator<DataSet>(trainTest.getTrain().asList(), nbatch);
		// System.out.println("Number of examples: " + dsiterTr.numExamples());
		// System.out.println("Number of labels: " + dsiterTr.getLabels());
		// System.out.println("Training on " + dsiterTr.getLabels()); // this might
		// return null

		ListDataSetIterator<DataSet> dsiterTe = new ListDataSetIterator<DataSet>(trainTest.getTest().asList(), nbatch);
		MultipleEpochsIterator epochitTr = new MultipleEpochsIterator(nepochs, dsiterTr);
		MultipleEpochsIterator epochitTe = new MultipleEpochsIterator(nepochs, dsiterTe);

		ConvolutionLayer layer_0 = new ConvolutionLayer.Builder(6, 6)
				.nIn(nChannels)
				.stride(2, 2) // default stride(2,2)
				.nOut(20) // # of feature maps
				.dropOut(0.7) // dropout to reduce overfitting
				.activation(Activation.RELU) // Activation: rectified linear units
				.build();

		SubsamplingLayer layer_1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
				.stride(2, 2)
				.build();

		ConvolutionLayer layer_2 = new ConvolutionLayer.Builder(6, 6)
				.stride(2, 2) // nIn need not specified in later layers
				.nOut(50)
				.activation(Activation.RELU) // Activation: rectified linear units
				.build();

		SubsamplingLayer layer_3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
				.stride(2, 2)
				.build();

		DenseLayer layer_4 = new DenseLayer.Builder() // Fully connected layer
				.nOut(500)
				.dropOut(0.7) // dropout to reduce overfitting
				.activation(Activation.RELU) // Activation: rectified linear units
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(10)
				.build();

		OutputLayer layer_5 = new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
				.nOut(outputNum) // number of classes to be predicted
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
				.gradientNormalizationThreshold(10)
				.activation(Activation.SOFTMAX)
				.build();

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).miniBatch(true)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l2(0.001) // l2 regularization on
																								// all layers
				.updater(new AdaGrad(0.001)).weightInit(WeightInit.XAVIER) // Xavier weight init
				.list().layer(0, layer_0).layer(1, layer_1).layer(2, layer_2).layer(3, layer_3).layer(4, layer_4)
				.layer(5, layer_5).setInputType(InputType.convolutionalFlat(numRows, numColumns, nChannels))
				.backprop(true).pretrain(false) // Feedforward hence no pre-train.
				.build();

		// log.info("Build model....");
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(Collections.singletonList(new ScoreIterationListener(listenerFreq)));

		// Print the number of parameters in the network (and for each layer)
		Layer[] layers = model.getLayers();
		int totalNumParams = 0;
		for (int i = 0; i < layers.length; i++) {
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

		// Initialize the user interface backend
		UIServer uiServer = UIServer.getInstance();

		// Configure where the network information (gradients, activations, score vs.
		// time etc) is to be stored
		// Then add the StatsListener to collect this information from the network, as
		// it trains
		StatsStorage statsStorage = new InMemoryStatsStorage(); // Alternative: new FileStatsStorage(File) - see
																// UIStorageExample

		// Attach the StatsStorage instance to the UI: this allows the contents of the
		// StatsStorage to be visualized
		uiServer.attach(statsStorage);

		int listenerFrequency = 1;
		model.setListeners(new StatsListener(statsStorage, listenerFrequency));

		// model.fit(epochitTr);
		log.info("Train model....");
		for (int i = 0; i < nepochs; i++) {
			model.fit(epochitTr);
		}

		// I think this could be done without an iterator and batches.
		log.info("Evaluate model....");
		// System.out.println("Testing on ...");
		Evaluation eval = new Evaluation(outputNum);
		while (epochitTe.hasNext()) {
			DataSet testDS = epochitTe.next(nbatch);
			INDArray output = model.output(testDS.getFeatureMatrix());
			eval.eval(testDS.getLabels(), output);
		}

		System.out.println(eval.stats());

		if (!saveNN.isEmpty()) {
			// model config
			FileUtils.write(new File(saveNN + ".json"), model.getLayerWiseConfigurations().toJson());

			// model parameters
			DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(saveNN + ".bin")));
			Nd4j.write(model.params(), dos);
		}

		log.info("****************Example finished********************");
	}
}
