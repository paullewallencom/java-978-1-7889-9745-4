package com.packt.JavaDL.MovieRecommendation.GraphUtil;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class PlotUtil_Rating {
	public static void plot(double[] iterationArray, double[] testArray, double[] trainArray) {
	final XYSeriesCollection dataSet = new XYSeriesCollection();
	addSeries(dataSet, iterationArray, testArray, "Test MSE per iteration");
	addSeries(dataSet, iterationArray, trainArray, "Training MSE per iteration");
	
	final JFreeChart chart = ChartFactory.createXYLineChart(
			"Training and Test error/iteration (1000 iterations)", // chart title
			"Iteration", // x axis label
			"MSE", // y axis label
			dataSet, // data
			PlotOrientation.VERTICAL,
			true, // include legend
			true, // tooltips
			false // urls
	);
	
	final ChartPanel panel = new ChartPanel(chart);
	final JFrame f = new JFrame();
	f.add(panel);
	f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
	f.pack();
	f.setVisible(true);
}

private static void addSeries (final XYSeriesCollection dataSet, double[] x, double[] y, final String label){
	final XYSeries s = new XYSeries(label);
	for( int j = 0; j < x.length; j++ ) s.add(x[j], y[j]);
	dataSet.addSeries(s);
}}