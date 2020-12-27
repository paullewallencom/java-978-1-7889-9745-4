package com.packt.JavaDL.Titanic;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationAveraging;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.deeplearning4j.nn.layers.normalization.BatchNormalization;

public class TitanicSurvivalPrediction {
	private static final Logger log = LoggerFactory.getLogger(TitanicSurvivalPrediction.class);	

	private static DataSetIterator readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses) throws IOException, InterruptedException {
		RecordReader rr = new CSVRecordReader();
		File input = new File(csvFileClasspath);
		rr.initialize(new FileSplit(input));
		DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
		return iterator;
	}

	public static void main(String[] args) throws Exception {
		String trainPath = "data/Titanic_Train.csv";
		String testPath = "data/Titanic_Test.csv";

		int labelIndex = 7; // First 7 features are followed by the labels in integer 
		int numClasses = 2; // number of classes to be predicted -i.e survived or not-survived
		int numEpochs = 10000; // Number of training eopich
	
		int seed = 123; // Randome seed for reproducibilty
		int numInputs = labelIndex; // Number of inputs in input layer
		int numOutputs = numClasses; // Number of classes to be predicted by the network 
		
		int batchSizeTraining = 128; 		
		// this is the data we want to use for training 
		DataSetIterator trainingDataIt = readCSVDataset(trainPath, batchSizeTraining, labelIndex, numClasses);
		
        // Data normalization 
		NormalizerMinMaxScaler preProcessor = new NormalizerMinMaxScaler();
	    preProcessor.fit(trainingDataIt);
	    trainingDataIt.setPreProcessor(preProcessor);
	    
		// this is the data we want to classify
		int batchSizeTest = 128;
		DataSetIterator testDataIt = readCSVDataset(testPath, batchSizeTest, labelIndex, numClasses);
		testDataIt.setPreProcessor(preProcessor); // same normalization for better results
		
		//Create the network layers. We have 4 layers. The first layer is the input layer, then 2 layers are placed as hidden layers. 
		// For the first 3 layers, we initialized the weights using Xavier and the activation function is ReLU. Finally, the output layer is followed by.
		
		// Input layer: we have specified neurons that are equal number of inputs and an arbitrary number of neurons as output. We set a smaller value considering a very few inputs and features. 
		DenseLayer input_layer = new DenseLayer.Builder()
				.weightInit(WeightInit.XAVIER)
				.activation(Activation.RELU)
				.nIn(numInputs).nOut(16)
				.build();
		
		// Hidden layer 1: number of input neurons is equal to the output of the input layer. Then the number of output is an arbitrary value. We set a smaller value considering a very few inputs and features. 
		DenseLayer hidden_layer_1 = new DenseLayer.Builder()
				.weightInit(WeightInit.XAVIER)
				.activation(Activation.RELU)
				.nIn(16).nOut(32)
				.build();
		
		// Hidden layer 2: number of input neurons is equal to the output of the hidden layer 1. Then the number of output is an arbitrary value. We set a smaller value considering a very few inputs and features. 
		DenseLayer hidden_layer_2 = new DenseLayer.Builder()
				.weightInit(WeightInit.XAVIER)
				.activation(Activation.RELU)
				.nIn(32).nOut(16)
				.build();
		
		// Output layer: number of input neurons is equal to the output of the hidden layer 1. Then the number of output is equal to number of predicted labels.  
		// We set a smaller value considering a very few inputs and features. 
		// Here we used Softmax is used as the activation function and the loss function as Cross Entropy since we want to convert the output (i.e. probability) to discrete class (i.e. 0 or 1) 
		OutputLayer output_layer = new OutputLayer.Builder(LossFunction.XENT) // XENT: Cross Entropy: Binary Classification
				.weightInit(WeightInit.XAVIER)
				.activation(Activation.SOFTMAX)
				.nIn(16).nOut(numOutputs)
				.build();
		
		// Create network configuration and conduct network training
		MultiLayerConfiguration MLPconf = new NeuralNetConfiguration.Builder().seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.weightInit(WeightInit.XAVIER)
				.updater(new Adam(0.0001))
				.list()
					.layer(0, input_layer)
					.layer(1, hidden_layer_1)
					.layer(2, hidden_layer_2)
					.layer(3, output_layer)
				.pretrain(false).backprop(true).build();			

		MultiLayerNetwork model = new MultiLayerNetwork(MLPconf);
        model.init();
        
      //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        
        int listenerFrequency = 1;
        model.setListeners(new StatsListener(statsStorage, listenerFrequency));


        log.info("Train model....");
        for( int i=0; i<numEpochs; i++ ){
            model.fit(trainingDataIt);
        }

        log.info("Evaluate model....");
        System.out.println("Evaluate model....");
        Evaluation eval = new Evaluation(2); // specify Evaluation(2, 0) for 0 class
        while(testDataIt.hasNext()){
            DataSet test = testDataIt.next();
            INDArray features = test.getFeatureMatrix();
            INDArray lables = test.getLabels();
            INDArray predicted = model.output(features,false);

            eval.eval(lables, predicted);

        }

        //Print the evaluation statistics
        System.out.println(eval.stats());
        
        // Compute Matthews correlation coefficient
        EvaluationAveraging averaging = EvaluationAveraging.Macro;
        double MCC = eval.matthewsCorrelation(averaging);
        System.out.println("Matthews correlation coefficient: "+ MCC);
        
        log.info("****************Example finished********************");      
        
	}
}