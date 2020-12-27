package com.packt.JavaDL.ReinforcementLearning;

import java.util.Random;
import java.util.Scanner;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/*
 * @author: Md. Rezaul Karim, 06/09/2018
 * 
 */

public class GridWorld {
	DeepQNetwork RLNet;
	int size = 4;
	//int scale = 3;

	float FrameBuffer[][];

	// Network initialization 
	void networkConstruction() {
		int InputLength = size * size * 2 + 1;
		int HiddenLayerCount = 150;
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .seed(12345)    //Random number generator seed for improved repeatability. Optional.
	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .weightInit(WeightInit.XAVIER)
	            .updater(new Adam(0.001))
	            .l2(0.001) // l2 regularization on all layers
	            .list()
				.layer(0, new DenseLayer.Builder()
						.nIn(InputLength)
						.nOut(HiddenLayerCount)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.build())
				.layer(1, new DenseLayer.Builder()
						.nIn(HiddenLayerCount)
						.nOut(HiddenLayerCount)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.build())
				.layer(2,new OutputLayer.Builder(LossFunction.MSE)
						.nIn(HiddenLayerCount)
						.nOut(4) // for 4 possible actions
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.IDENTITY)
						.weightInit(WeightInit.XAVIER)
						.build())
				.pretrain(false).backprop(true).build();

		RLNet = new DeepQNetwork(conf, 100000, .99f, 1d, 1024, 500, 1024, InputLength, 4);
	}

	Random rand = new Random();

	// Generate the GridMap
	float[][] generateGridMap() {
		int agent = rand.nextInt(size * size);
		int goal = rand.nextInt(size * size);
		while (goal == agent)
			goal = rand.nextInt(size * size);
		float[][] map = new float[size][size];
		for (int i = 0; i < size * size; i++)
			map[i / size][i % size] = 0;
		map[goal / size][goal % size] = -1;
		map[agent / size][agent % size] = 1;
		return map;
	}

	// Calculate the position of agent
	int calcAgentPos(float[][] Map) {
		int x = -1;
		for (int i = 0; i < size * size; i++) {
			if (Map[i / size][i % size] == 1)
				return i;
		}
		return x;
	}

	// Calculate the position of goal 
	int calcGoalPos(float[][] Map) {
		int x = -1;
		for (int i = 0; i < size * size; i++) {
			if (Map[i / size][i % size] == -1)
				return i;
		}
		return x;
	}

	// Get action mask
	int[] getActionMask(float[][] CurrMap) {
		int retVal[] = { 1, 1, 1, 1 };

		int agent = calcAgentPos(CurrMap);
		if (agent < size)
			retVal[0] = 0;
		if (agent >= size * size - size)
			retVal[1] = 0;
		if (agent % size == 0)
			retVal[2] = 0;
		if (agent % size == size - 1)
			retVal[3] = 0;

		return retVal;
	}

	// Show guidance move to agent 
	float[][] doMove(float[][] CurrMap, int action) {
		float nextMap[][] = new float[size][size];
		for (int i = 0; i < size * size; i++)
			nextMap[i / size][i % size] = CurrMap[i / size][i % size];

		int agent = calcAgentPos(CurrMap);
		nextMap[agent / size][agent % size] = 0;
		
		if (action == 0) {
			if (agent - size >= 0)
				nextMap[(agent - size) / size][agent % size] = 1;
			else {
				System.out.println("Bad Move");
				System.exit(0);
			}
		} else if (action == 1) {
			if (agent + size < size * size)
				nextMap[(agent + size) / size][agent % size] = 1;
			else {
				System.out.println("Bad Move");
				System.exit(0);
			}
		} else if (action == 2) {
			if ((agent % size) - 1 >= 0)
				nextMap[agent / size][(agent % size) - 1] = 1;
			else {
				System.out.println("Bad Move");
				System.exit(0);
			}
		} else if (action == 3) {
			if ((agent % size) + 1 < size)
				nextMap[agent / size][(agent % size) + 1] = 1;
			else {
				System.out.println("Bad Move");
				System.exit(0);
			}
		}
		return nextMap;
	}

	// Compute reward for an action 
	float calcReward(float[][] CurrMap, float[][] NextMap) {
		int newGoal = calcGoalPos(NextMap);

		if (newGoal == -1)
			return size * size + 1;

		return -1f;
	}

	void addToBuffer(float[][] nextFrame) {
		FrameBuffer = nextFrame;
	}

	INDArray flattenInput(int TimeStep) {
		float flattenedInput[] = new float[size * size * 2 + 1];
		for (int a = 0; a < size; a++) {
			for (int b = 0; b < size; b++) {
				if (FrameBuffer[a][b] == -1)
					flattenedInput[a * size + b] = 1;
				else
					flattenedInput[a * size + b] = 0;
				if (FrameBuffer[a][b] == 1)
					flattenedInput[size * size + a * size + b] = 1;
				else
					flattenedInput[size * size + a * size + b] = 0;
			}
		}
		flattenedInput[size * size * 2] = TimeStep;
		return Nd4j.create(flattenedInput);
	}

	void printGrid(float[][] Map) {
		for (int x = 0; x < size; x++) {
			for (int y = 0; y < size; y++) {
				System.out.print((int) Map[x][y]);
			}
			System.out.println(" ");
		}
		System.out.println(" ");
	}

	public static void main(String[] args) {
		GridWorld grid = new GridWorld();
		grid.networkConstruction();

		for (int m = 0; m < 10; m++) {
			System.out.println("Episode: " + m);
			float CurrMap[][] = grid.generateGridMap();

			grid.FrameBuffer = CurrMap;
			int t = 0;
			grid.printGrid(CurrMap);

			for (int i = 0; i < 2 * grid.size; i++) {
				int a = grid.RLNet.getAction(grid.flattenInput(t), grid.getActionMask(CurrMap));
				
				float NextMap[][] = grid.doMove(CurrMap, a);
				float r = grid.calcReward(CurrMap, NextMap);
				grid.addToBuffer(NextMap);
				t++;

				if (r == grid.size * grid.size + 1) {
					grid.RLNet.observeReward(r, null, grid.getActionMask(NextMap));
					break;
				}

				grid.RLNet.observeReward(r, grid.flattenInput(t), grid.getActionMask(NextMap));
				CurrMap = NextMap;
			}
		}

		// Reward calculation 
		//Scanner keyboard = new Scanner(System.in);
		float tReward = 0; // initially no reward
		for (int m = 0; m < 10; m++) {
			grid.RLNet.SetEpsilon(0);
			float CurrMap[][] = grid.generateGridMap();
			grid.FrameBuffer = CurrMap;

			int t = 0;

			//while (true) {
				grid.printGrid(CurrMap);
				//keyboard.nextLine();

				int a = grid.RLNet.getAction(grid.flattenInput(t), grid.getActionMask(CurrMap));
				float NextMap[][] = grid.doMove(CurrMap, a);
				float r = grid.calcReward(CurrMap, NextMap);

				tReward += r;
				grid.addToBuffer(NextMap);
				t++;
				grid.RLNet.observeReward(r, grid.flattenInput(t), grid.getActionMask(NextMap));

				if (r == grid.size * grid.size + 1) { // reached the goal{
					CurrMap = NextMap;
					System.out.println("Net Score: " + (tReward));
					break;
					}
				//CurrMap = NextMap;
			//}
			//System.out.println("Net Score: " + (tReward));
			
		}
		//System.out.println("Net Score: " + (tReward));
		//keyboard.close();
	}
}
