package com.packt.JavaDL.MovieRecommendation.Prediction;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

import com.packt.JavaDL.MovieRecommendation.DataUtils.DataProvider;
import com.packt.JavaDL.MovieRecommendation.DataUtils.LibSVMDataProvider;
import com.packt.JavaDL.MovieRecommendation.FMCore.DataMetaInfo;
import com.packt.JavaDL.MovieRecommendation.FMCore.FmLearn;
import com.packt.JavaDL.MovieRecommendation.FMCore.FmLearnSgd;
import com.packt.JavaDL.MovieRecommendation.FMCore.FmLearnSgdElement;
import com.packt.JavaDL.MovieRecommendation.FMCore.FmModel;
import com.packt.JavaDL.MovieRecommendation.Tools.Constants;
import com.packt.JavaDL.MovieRecommendation.Tools.Debug;
import com.packt.JavaDL.MovieRecommendation.Tools.RLog;
import com.packt.JavaDL.MovieRecommendation.Tools.TaskType;
import com.packt.JavaDL.MovieRecommendation.Tools.Util;

/*
 * @author Md. Rezaul Karim, 07/06/2018
 */

public class MovieRatingPrediction {
	private static String formattedDataPath = "formatted_data";
	private static String outPut = "outFolder";
	
	static public Integer[] getIntegerValues(String parameter) {
		Integer[] result = null;
		String[] strresult = Util.tokenize(parameter, ",");
		if(strresult!=null && strresult.length>0) {
			result = new Integer[strresult.length];
			for(int i=0;i<strresult.length;i++) {
				result[i] = Integer.parseInt(strresult[i]);
			}
		}
		return result;
	}

	static public Double[] getDoubleValues(String parameter) {
		Double[] result;
		String[] strresult = Util.tokenize(parameter, ",");
		if(strresult!=null && strresult.length>0) {
			result = new Double[strresult.length];
			for(int i=0; i<strresult.length; i++) {
				result[i] = Double.parseDouble(strresult[i]);
			}
		}
		else {
			result = new Double[0];
		}
		return result;
	}

	@SuppressWarnings("resource")
	public static void main(String[] args) throws Exception {
		// filename for training data
		final String trainFile = formattedDataPath+ "/" + "ratings_train.libfm";
		
		// filename for testing data
		final String testFile = formattedDataPath+ "/" + "ratings_test.libfm";
		
		// filename for testing meta data
		final String testMetaFile = formattedDataPath+ "/" + "ratings_test.libfm.meta";
		
		// filename for the final prediction output file
		final String outputFile = "outFolder" + "/" + "predict_output.txt";
		
		// setup the dimension: k0,k1,k2: k0=use bias, k1=use 1-way interactions, k2=dim of 2-way interactions
		final String dimension = "1,1,8";
		
		// number of iterations
		final String iterations = "10"; // tunable param
		
		// learning rate for SGD
		final String learnRate = "0.01";
		
		// setup regularization: r0,r1,r2: r0=bias regularization, r1=1-way regularization, r2=2-way regularization
		final String regularization = "0,0,0.1";
		
		// standard deviations for initialization of 2-way factors
		final String stdDeviation = "0.1";
		
		// write measurements within iterations to a file
		final String rLog = outPut + "/" + "metrics_logs.txt";

		// Load the data
		System.out.println("Loading train...\t");
		DataProvider train = new LibSVMDataProvider();
		Properties trainproperties = new Properties();
		
		trainproperties.put(Constants.FILENAME, trainFile);
		train.load(trainproperties,false);

		System.out.println("Loading test... \t");
		DataProvider test = new LibSVMDataProvider();
		Properties testproperties = new Properties();
		
		testproperties.put(Constants.FILENAME, testFile);
		test.load(testproperties,false);

		// (main table)
		int num_all_attribute = Math.max(train.getFeaturenumber(), test.getFeaturenumber());
		DataMetaInfo meta = new DataMetaInfo(num_all_attribute);
		meta.debug();
		Debug.openConsole();

		// (2) Setup the factorization machine
		FmModel fm = new FmModel();
		fm.num_attribute = num_all_attribute;
		fm.initstdev = Double.parseDouble(stdDeviation);
		
		// set the number of dimensions in the factorization
		Integer[] dim = getIntegerValues(dimension);
		assert (dim.length == 3);
		fm.k0 = dim[0] != 0;
		fm.k1 = dim[1] != 0;
		fm.num_factor = dim[2];
		fm.init();

		// (3) Setup the learning method: SGD
		FmLearn fml = new FmLearnSgdElement();
		((FmLearnSgd) fml).num_iter = Integer.parseInt(iterations);

		fml.fm = fm;
		fml.max_target = train.getMaxtarget();
		fml.min_target = train.getMintarget();
		fml.meta = meta;
		fml.task = TaskType.TASK_REGRESSION;

		// (4) init the logging
		System.out.println("logging to " + rLog);
		RLog rlog = new RLog(rLog);
		fml.log = rlog;
		fml.init();
		rlog.init();
		fm.debug();
		fml.debug();

		// set the regularization
		Double[] reg = getDoubleValues(regularization);
		assert ((reg.length == 3));
		fm.reg0 = reg[0];
		fm.regw = reg[1];
		fm.regv = reg[2];

		FmLearnSgd fmlsgd = (FmLearnSgd) (fml);
		if (fmlsgd != null) {
			// set the learning rates (individual per layer)
			Double[] lr = getDoubleValues(learnRate);
			assert (lr.length == 1);
			fmlsgd.learn_rate = lr[0];
			Arrays.fill(fmlsgd.learn_rates, lr[0]);
		}

		// learn the model
		fml.learn(train, test);

		// prediction at the end
		String print = String.format("#Iterations=%s::  Train_RMSE=%-10.5f  Test_RMSE=%-10.5f", iterations, fml.evaluate(train), fml.evaluate(test));
		System.out.println(print);

		// save prediction
		Map<Integer, String> ratingsMetaData = new HashMap<>();
		if (Files.exists(Paths.get(testMetaFile))) {
			BufferedReader bufferedReader = new BufferedReader(new FileReader(testMetaFile));
			String line;
			
			while ((line = bufferedReader.readLine()) != null) {
				String[] splitLine = line.split("\\s+");
				if (splitLine.length > 0) {
					Integer indexKey = Integer.parseInt(splitLine[2]);
					String userIdmovieIdValue = splitLine[0] + " " +  splitLine[1];
					ratingsMetaData.put(indexKey, userIdmovieIdValue);
				}
			}
		}

		double[] pred = new double[test.getRownumber()];
		fml.predict(test, pred);
		Util.save(ratingsMetaData, pred, outputFile);
		
		String FILENAME = Constants.FILENAME;
		// Save the trained FM model 
		fmlsgd.saveModel(FILENAME);
	}
}
