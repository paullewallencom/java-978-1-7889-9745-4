package com.packt.JavaDL.MovieRecommendation.FMCore;

import java.util.ArrayList;
import java.util.List;

import com.packt.JavaDL.MovieRecommendation.DataUtils.DataProvider;
import com.packt.JavaDL.MovieRecommendation.GraphUtil.PlotUtil_Rating;
import com.packt.JavaDL.MovieRecommendation.Tools.Debug;
import com.packt.JavaDL.MovieRecommendation.Tools.JlibfmRuntimeException;
import com.packt.JavaDL.MovieRecommendation.Tools.TaskType;
import com.packt.JavaDL.MovieRecommendation.Tools.Util;

public class FmLearnSgdElement extends FmLearnSgd {	
	public void init() {
		super.init();
		if (log != null) {
			log.addField("rmse_train", Double.NaN);
		}
	}
	
	public void learn(DataProvider train, DataProvider test)  throws Exception{
		super.learn(train, test);
		List<Double> iterationList=new ArrayList<Double>();
        List<Double> trainList=new ArrayList<Double>();
        List<Double> testList=new ArrayList<Double>();
		// Debug.println("SGD: DON'T FORGET TO SHUFFLE THE ROWS IN TRAINING DATA TO GET THE BEST RESULTS.");
		// SGD
		for (int i = 0; i < num_iter; i++) {
			try
			{
				double iteration_time = Util.getusertime();
				train.shuffle();
				for (train.getData().begin(); !train.getData().end(); train.getData().next()) {
					double p = fm.predict(train.getData().getRow(), sum, sum_sqr);
					double mult = 0;
					if (task == TaskType.TASK_REGRESSION) {
						p = Math.min(max_target, p);
						p = Math.max(min_target, p);
						mult = -(train.getTarget()[train.getData().getRowIndex()]-p);
					} else if (task == TaskType.TASK_CLASSIFICATION) {
						mult = -train.getTarget()[train.getData().getRowIndex()]*(1.0-1.0/(1.0+Math.exp(-train.getTarget()[train.getData().getRowIndex()]*p)));
					}				
					SGD(train.getData().getRow(), mult, sum);					
				}				
				iteration_time = (Util.getusertime() - iteration_time);
				double rmse_train = evaluate(train);
				double rmse_test = evaluate(test);
				iterationList.add((double)i);
				testList.add(rmse_test);
				trainList.add(rmse_train);
				
				String print = String.format("#Iterations=%2d::  Train_RMSE=%-10.5f  Test_RMSE=%-10.5f", i, rmse_train, rmse_test);
				Debug.println(print);
				if (log != null) {
					log.log("rmse_train", rmse_train);
					log.log("time_learn", iteration_time);
					log.newLine();
				}
			}
			catch(Exception e)
			{
				throw new JlibfmRuntimeException(e);
			}
		}	
		PlotUtil_Rating.plot(convertobjectArraytoDouble(iterationList.toArray()),
        		convertobjectArraytoDouble(testList.toArray()),
        		convertobjectArraytoDouble(trainList.toArray()));

	}
	 public double [] convertobjectArraytoDouble(Object[] objectArray){
		   double[] doubleArray = new double[objectArray.length];
		   //Double[ ]doubleArray=new Double();
		   for(int i = 0; i < objectArray.length; i++){
		  		Object object = objectArray[i];
		  		String string = object.toString(); double dub = Double.valueOf(string).doubleValue();
		  		doubleArray[i] = dub;
		  		
		  	}
		   return doubleArray;
	    }	
}
