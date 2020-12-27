package com.packt.JavaDL.CancerPrediction_TCGA;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CancerPreddictionLSTM {
	private static final Logger log = LoggerFactory.getLogger(CancerPreddictionLSTM.class);
	static int batchSizePerWorker = 16;
	static int numEpochs = 10;

	private static DataSetIterator readCSVDataset(String csvFileClasspath, int batchSize, int labelIndex, int numClasses) throws IOException, InterruptedException {
		RecordReader rr = new CSVRecordReader();
		File input = new File(csvFileClasspath);
		rr.initialize(new FileSplit(input));
		DataSetIterator iterator = new RecordReaderDataSetIterator(rr, batchSize, labelIndex, numClasses);
		return iterator;
	}

	public static void main(String[] args) throws Exception {	

		// Show data paths
		String trainPath = "C:/Users/admin-karim/Desktop/TCGA-PANCAN/TCGA_train.csv";
		String testPath = "C:/Users/admin-karim/Desktop/TCGA-PANCAN/TCGA_test.csv";	
		
		// ----------------------------------
		// Preparing training and test set. 	
		int labelIndex = 20531;			
		int numClasses = 5; 
		int batchSize = 128; 
		
		// This dataset is used for training 
		DataSetIterator trainingDataIt = readCSVDataset(trainPath, batchSize, labelIndex, numClasses);

		// This is the data we want to classify
		DataSetIterator testDataIt = readCSVDataset(testPath, batchSize, labelIndex, numClasses);		
		
		// ----------------------------------
		// Network hyperparameters
		int seed = 12345;
		int numInputs = labelIndex;
		int numOutputs = numClasses;		
		int numHiddenNodes = 5000;
		
		// Create network configuration and conduct network training
        MultiLayerConfiguration LSTMconf = new NeuralNetConfiguration.Builder()
            .seed(seed)    //Random number generator seed for improved repeatability. Optional.
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.001))
            .list()
            .layer(0, new LSTM.Builder()
            			.nIn(numInputs)
            			.nOut(numHiddenNodes)
            			.activation(Activation.RELU)
            			.build())
            .layer(1, new LSTM.Builder()
            			.nIn(numHiddenNodes)
            			.nOut(numHiddenNodes)
            			.activation(Activation.RELU)
            			.build())
            .layer(2, new LSTM.Builder()
            			.nIn(numHiddenNodes)
            			.nOut(numHiddenNodes)
            			.activation(Activation.RELU)
            			.build())
            .layer(3, new RnnOutputLayer.Builder()
            			.activation(Activation.SOFTMAX)
            			.lossFunction(LossFunction.MCXENT)
            			.nIn(numHiddenNodes)
            			.nOut(numOutputs)
            			.build())
            .pretrain(false).backprop(true).build();

        // Create and initialize multilayer network 
		MultiLayerNetwork model = new MultiLayerNetwork(LSTMconf);
        model.init();
        
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = model.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);
		
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
        Evaluation eval = new Evaluation(5); //create an evaluation object with 10 possible classes
        while(testDataIt.hasNext()){
            DataSet next = testDataIt.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");
	}
}