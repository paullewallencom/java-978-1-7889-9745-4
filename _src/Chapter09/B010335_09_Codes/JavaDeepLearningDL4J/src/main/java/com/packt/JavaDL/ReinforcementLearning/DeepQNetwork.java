package com.packt.JavaDL.ReinforcementLearning;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
/*
 * @author: Md. Rezaul Karim, 06/09/2018
 * 
 */

public class DeepQNetwork {
	int ReplayMemoryCapacity;
	List<Replay> ReplayMemory;
	double Epsilon;
	float Discount;
	
	MultiLayerNetwork DeepQ;
	MultiLayerNetwork TargetDeepQ;
	
	int BatchSize;
	int UpdateFreq;
	int UpdateCounter;
	int ReplayStartSize;
	Random r;
	
	int InputLength;
	int NumActions;
	
	INDArray LastInput;
	int LastAction;
	
	DeepQNetwork(MultiLayerConfiguration conf, int replayMemoryCapacity, float discount, double epsilon, int batchSize, int updateFreq, int replayStartSize, int inputLength, int numActions){
		DeepQ = new MultiLayerNetwork(conf);
		DeepQ.init();
		
		TargetDeepQ = new MultiLayerNetwork(conf);
		TargetDeepQ.init();
		
		TargetDeepQ.setParams(DeepQ.params());
		ReplayMemoryCapacity = replayMemoryCapacity;
		
		Epsilon = epsilon;
		Discount = discount;
		
		r = new Random();
		BatchSize = batchSize;
		UpdateFreq = updateFreq;
		UpdateCounter = 0;
		
		ReplayMemory = new ArrayList<Replay>();
		ReplayStartSize = replayStartSize;
		InputLength = inputLength;
		NumActions = numActions;
	}
	
	void SetEpsilon(double e){
		Epsilon = e;
	}
	
// We first run our agent to collect enough transitions to fill up the replay memory, without training. For example, our memory may be of size 10,000.
//Then at every step, agent will obtain a transition and we add this to the end of the memory, and pop off the earliest one. 
//Then sample a mini batch of experiences from the memory randomly, and update our Q function on that, similar to mini-batch gradient descent. 
	void addReplay(float reward , INDArray NextInput , int NextActionMask[]){
		if( ReplayMemory.size() >= ReplayMemoryCapacity )
			ReplayMemory.remove( r.nextInt(ReplayMemory.size()) );
		
		ReplayMemory.add(new Replay(LastInput , LastAction , reward , NextInput , NextActionMask));
	}
	
	Replay[] getMiniBatch(int BatchSize){
		int size = ReplayMemory.size() < BatchSize ? ReplayMemory.size() : BatchSize ;
		Replay[] retVal = new Replay[size];
		
		for(int i = 0 ; i < size ; i++){
			retVal[i] = ReplayMemory.get(r.nextInt(ReplayMemory.size()));
		}
		return retVal;
		
	}
	
	float findMax(INDArray NetOutputs , int ActionMask[]){
		int i = 0;
		while(ActionMask[i] == 0) i++;
		
		float maxVal = NetOutputs.getFloat(i);
		for(; i < NetOutputs.size(1) ; i++){
			if(NetOutputs.getFloat(i) > maxVal && ActionMask[i] == 1){
				maxVal = NetOutputs.getFloat(i);
			}
		}
		return maxVal;
	}
	
	int findActionMax(INDArray NetOutputs , int ActionMask[]){
		int i = 0;
		while(ActionMask[i] == 0) i++;
		
		float maxVal = NetOutputs.getFloat(i);
		int maxValI = i;
		for(; i < NetOutputs.size(1) ; i++){
			if(NetOutputs.getFloat(i) > maxVal && ActionMask[i] == 1){
				maxVal = NetOutputs.getFloat(i);
				maxValI = i;
			}
		}
		return maxValI;
	}	
	
	int getAction(INDArray Inputs , int ActionMask[]){
		LastInput = Inputs;
		INDArray outputs = DeepQ.output(Inputs);
		
		System.out.print(outputs + " ");
		if(Epsilon > r.nextDouble()) {
			 LastAction = r.nextInt(outputs.size(1));
			 while(ActionMask[LastAction] == 0)
				 LastAction = r.nextInt(outputs.size(1));
			 System.out.println(LastAction);
			 return LastAction;
		}
		
		LastAction = findActionMax(outputs , ActionMask);
		System.out.println(LastAction);
		return LastAction;
	}
	
	void observeReward(float Reward , INDArray NextInputs , int NextActionMask[]){
		addReplay(Reward , NextInputs , NextActionMask);
		if(ReplayStartSize <  ReplayMemory.size())
			networkTraining(BatchSize);
		UpdateCounter++;
		if(UpdateCounter == UpdateFreq){
			UpdateCounter = 0;
			System.out.println("Reconciling Networks");
			reconcileNetworks();
		}
	}
	
	INDArray combineInputs(Replay replays[]){
		INDArray retVal = Nd4j.create(replays.length , InputLength);
		for(int i = 0; i < replays.length ; i++){
			retVal.putRow(i, replays[i].Input);
		}
		return retVal;
	}
	
	INDArray combineNextInputs(Replay replays[]){
		INDArray retVal = Nd4j.create(replays.length , InputLength);
		for(int i = 0; i < replays.length ; i++){
			if(replays[i].NextInput != null)
				retVal.putRow(i, replays[i].NextInput);
		}
		return retVal;
	}
	
	void networkTraining(int BatchSize){
		Replay replays[] = getMiniBatch(BatchSize);
		INDArray CurrInputs = combineInputs(replays);
		INDArray TargetInputs = combineNextInputs(replays);

		INDArray CurrOutputs = DeepQ.output(CurrInputs);
		INDArray TargetOutputs = TargetDeepQ.output(TargetInputs);
		
		float y[] = new float[replays.length];
		for(int i = 0 ; i < y.length ; i++){
			int ind[] = { i , replays[i].Action };
			float FutureReward = 0 ;
			if(replays[i].NextInput != null)
				FutureReward = findMax(TargetOutputs.getRow(i) , replays[i].NextActionMask);
			float TargetReward = replays[i].Reward + Discount * FutureReward ;
			CurrOutputs.putScalar(ind , TargetReward ) ;
		}
		//System.out.println("Avgerage Error: " + (TotalError / y.length) );
		
		DeepQ.fit(CurrInputs, CurrOutputs);
	}
	
	void reconcileNetworks(){
		TargetDeepQ.setParams(DeepQ.params());
	}
	
	public boolean saveNetwork(String ParamFileName , String JSONFileName){
	    //Write the network parameters for later use:
	    try(DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(ParamFileName)))){
	        Nd4j.write(DeepQ.params(),dos);
	    } catch (IOException e) {
	    	System.out.println("Failed to write params");
			return false;
		}
	    
	    //Write the network configuration:
	    try {
			FileUtils.write(new File(JSONFileName), DeepQ.getLayerWiseConfigurations().toJson());
		} catch (IOException e) {
			System.out.println("Failed to write json");
			return false;
		}
	    return true;
	}
	
	public boolean restoreNetwork(String ParamFileName , String JSONFileName){
		//Load network configuration from disk:
	    MultiLayerConfiguration confFromJson;
		try {
			confFromJson = MultiLayerConfiguration.fromJson(FileUtils.readFileToString(new File(JSONFileName)));
		} catch (IOException e1) {
			System.out.println("Failed to load json");
			return false;
		}

	    //Load parameters from disk:
	    INDArray newParams;
	    try(DataInputStream dis = new DataInputStream(new FileInputStream(ParamFileName))){
	        newParams = Nd4j.read(dis);
	    } catch (FileNotFoundException e) {
	    	System.out.println("Failed to load parems");
			return false;
		} catch (IOException e) {
	    	System.out.println("Failed to load parems");
			return false;
		}
	    //Create a MultiLayerNetwork from the saved configuration and parameters 
	    DeepQ = new MultiLayerNetwork(confFromJson); 
	    DeepQ.init(); 
	    
	    DeepQ.setParameters(newParams); 
	    reconcileNetworks();
	    return true;	    
	}	
}
