package com.packt.JavaDL.MovieRecommendation.GraphUtil;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

public class PlotUtil_Rank {	
	public static void plot(double[] iterationArray, double[] timeArray, String chart_type, int iter) {
		String series = null;
		String title = null;
		String x_axis = null;
		String y_axis = null;
		
		if(chart_type =="MSE"){		
			series = "MSE per Iteration (" + iter + " iterations)";
			title = "MSE per Iteration (" + iter + " iterations)";
			x_axis = "Iteration";
			y_axis = "MSE";
		}else {
			series = "Time per Iteration (" + iter + " iterations)";
			title = "Time per Iteration (" + iter + " iterations)";
			x_axis = "Iteration";
			y_axis = "Time";			
		}
			final XYSeriesCollection dataSet = new XYSeriesCollection();
			addSeries(dataSet, iterationArray, timeArray, series);
		
			final JFreeChart chart = ChartFactory.createXYLineChart(
					title, // chart title
					x_axis, // x axis label
					y_axis, // y axis label
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
	}
}