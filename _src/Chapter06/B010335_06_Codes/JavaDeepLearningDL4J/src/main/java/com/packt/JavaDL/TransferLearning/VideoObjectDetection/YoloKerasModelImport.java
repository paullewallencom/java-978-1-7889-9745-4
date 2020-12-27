package com.packt.JavaDL.TransferLearning.VideoObjectDetection;

import java.io.IOException;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;

public class YoloKerasModelImport {
	public static int nBoxes = 5;
	public static double[][] priorBoxes = { { 1.08, 1.19 }, { 3.42, 4.41 }, { 6.63, 11.38 }, { 9.42, 5.11 },
			{ 16.62, 10.52 } };

	private static long seed;
	private static WorkspaceMode workspaceMode;

	public static void main(String[] args)
			throws IOException, UnsupportedKerasConfigurationException, InvalidKerasConfigurationException {
		String pretrainedModelPath = "bin/yolo.h5";
		//ComputationGraph network = KerasModelImport.importKerasModelAndWeights(pretrainedModelPath);

		//ModelSerializer.writeModel(network, "bin/Yolo_v3.zip", false);

		//String filename = "tiny-yolo-voc.h5";
		ComputationGraph graph = KerasModelImport.importKerasModelAndWeights(pretrainedModelPath, false);
		INDArray priors = Nd4j.create(priorBoxes);

		FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder().seed(seed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.gradientNormalization(GradientNormalization.RenormalizeL2PerLayer).gradientNormalizationThreshold(1.0)
				.updater(new Adam.Builder().learningRate(1e-3).build()).l2(0.00001).activation(Activation.IDENTITY)
				.trainingWorkspaceMode(workspaceMode).inferenceWorkspaceMode(workspaceMode).build();

		ComputationGraph model = new TransferLearning.GraphBuilder(graph).fineTuneConfiguration(fineTuneConf)
				.addLayer("outputs", new Yolo2OutputLayer.Builder().boundingBoxPriors(priors).build(), "conv2d_9")
				.setOutputs("outputs").build();

		System.out.println(model.summary(InputType.convolutional(416, 416, 3)));

		ModelSerializer.writeModel(model, "bin/Yolo_v3.zip", false);

	}

}
