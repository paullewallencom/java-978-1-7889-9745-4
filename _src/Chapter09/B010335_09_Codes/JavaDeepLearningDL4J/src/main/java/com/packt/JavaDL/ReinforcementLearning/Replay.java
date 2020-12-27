package com.packt.JavaDL.ReinforcementLearning;

import org.nd4j.linalg.api.ndarray.INDArray;
/*
 * @author: Md. Rezaul Karim, 06/09/2018
 * 
 */

public class Replay {
	INDArray Input;
	int Action; 
	float Reward;
	INDArray NextInput;
	int NextActionMask[] ;
	
	// Initialize Replay memory
	Replay(INDArray input , int action , float reward , INDArray nextInput , int nextActionMask[]){
		Input = input;
		Action = action;
		Reward = reward;
		NextInput = nextInput;
		NextActionMask = nextActionMask ;
	}
	
}
