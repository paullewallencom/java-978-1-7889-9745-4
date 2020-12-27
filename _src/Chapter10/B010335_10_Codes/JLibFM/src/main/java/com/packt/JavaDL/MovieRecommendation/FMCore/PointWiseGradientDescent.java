package com.packt.JavaDL.MovieRecommendation.FMCore;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;
import org.ranksys.javafm.FM;
import org.ranksys.javafm.learner.FMLearner;
import org.ranksys.javafm.learner.gd.PointWiseError;

import com.packt.JavaDL.MovieRecommendation.GraphUtil.PlotUtil_Rank;

import org.ranksys.javafm.data.FMData;

public class PointWiseGradientDescent implements FMLearner<FMData> {
    private static final Logger LOG = Logger.getLogger(PointWiseGradientDescent.class.getName());

    private final double learnRate;
    private final int numIter;
    private final PointWiseError error;
    private final double regB;
    private final double[] regW;
    private final double[] regM;
    private int iter = 0;

    public PointWiseGradientDescent(double learnRate, int numIter, PointWiseError error, double regB, double[] regW, double[] regM) {
        this.learnRate = learnRate;
        this.numIter = numIter;
        this.error = error;
        this.regB = regB;
        this.regW = regW;
        this.regM = regM;
    }

    @Override
    public double error(FM fm, FMData test) {
        return test.stream()
                .mapToDouble(x -> error.error(fm, x))
                .average().getAsDouble();
    }

    @Override
    public void learn(FM fm, FMData train, FMData test) {
        LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", 0, error(fm, train), error(fm, test)));
        List<Double> iterationList = new ArrayList<Double>();
        List<Double> timeList = new ArrayList<Double>();
        List<Double> errList = new ArrayList<Double>();
        
        for (int t = 1; t <= numIter; t++) {
            long time0 = System.nanoTime();

            train.shuffle();

            train.stream().forEach(x -> {
                double b = fm.getB();
                double[] w = fm.getW();
                double[][] m = fm.getM();

                double lambda = error.dError(fm, x);

                fm.setB(b - learnRate * (lambda + regB * b));

                double[] xm = new double[m[0].length];
                x.consume((i, xi) -> {
                    for (int j = 0; j < xm.length; j++) {
                        xm[j] += xi * m[i][j];
                    }

                    w[i] -= learnRate * (lambda * xi + regW[i] * w[i]);
                });

                x.consume((i, xi) -> {
                    for (int j = 0; j < m[i].length; j++) {
                        m[i][j] -= learnRate * (lambda * xi * xm[j]
                                - lambda * xi * xi * m[i][j]
                                + regM[i] * m[i][j]);
                    }
                });
            });

            iter = t;
            long time1 = System.nanoTime() - time0;
            iterationList.add((double)iter);
            timeList.add((double)time1 / 1_000_000_000.0);
            errList.add(error(fm, test));
            
           
            //PlotUtils.plot(iterationList.toArray(),timeList.toArray(),"test");
            LOG.info(String.format("iteration n = %3d t = %.2fs", iter, time1 / 1_000_000_000.0));
            LOG.fine(() -> String.format("iteration n = %3d e = %.6f e = %.6f", iter, error(fm, train), error(fm, test)));
        }
        
        PlotUtil_Rank.plot(convertobjectArraytoDouble(iterationList.toArray()),	convertobjectArraytoDouble(errList.toArray()), "MSE", iter);        
        PlotUtil_Rank.plot(convertobjectArraytoDouble(iterationList.toArray()), convertobjectArraytoDouble(timeList.toArray()), "TIME", iter);
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
