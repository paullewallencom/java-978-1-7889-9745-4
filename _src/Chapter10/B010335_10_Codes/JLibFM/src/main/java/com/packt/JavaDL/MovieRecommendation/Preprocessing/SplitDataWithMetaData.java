/*
 * JLibFM
 *
 * Copyright (c) 2017, Jinbo Chen(gaterslebenchen@gmail.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 *  - Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the docume
 *    ntation and/or other materials provided with the distribution.
 *  - Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUD
 * ING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN N
 * O EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR C
 * ONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR P
 * ROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 *  TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBI
 *  LITY OF SUCH DAMAGE.
 */
package com.packt.JavaDL.MovieRecommendation.Preprocessing;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.OutputStreamWriter;
import java.io.Reader;
import java.util.Random;

/*
 * @author: Md. Rezaul Karim. However, this an extended version of original example provided in GitHub
 */
 
/**
 * 
 * split data to training dataset, test dataset and validation dataset
 *
 */
public class SplitDataWithMetaData {
	private static String formattedDataPath = "formatted_data";
	private static String ratinglibFM = formattedDataPath + "/" + "ratings.libfm";
	private static String ratinglibFM_train = formattedDataPath + "/" + "ratings_train.libfm";
	private static String ratinglibFM_test = formattedDataPath + "/" + "ratings_test.libfm";
	private static String ratinglibFM_test_meta = formattedDataPath + "/" + "ratings_test.libfm.meta";	
	private static String ratinglibFM_valid = formattedDataPath + "/" + "ratings_valid.libfm";

	public static void main(String[] args) throws Exception {
		Reader fr = new FileReader(ratinglibFM);

		Random ra = new Random();
		BufferedWriter trainwrite = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(ratinglibFM_train)));
		BufferedWriter testwrite = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(ratinglibFM_test)));
		BufferedWriter testmetawrite = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(ratinglibFM_test_meta)));		
		BufferedWriter validwrite = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(ratinglibFM_valid)));

		BufferedReader br = new BufferedReader(fr);
		String line = null;
		int testline = 0;
		
		while (br.ready()) {
			line = br.readLine();
			String[] arr = line.split("#");
			String info = arr[0];
			
			double dvalue = ra.nextDouble();
			if(dvalue>0.9)
			{
				validwrite.write(info);
				validwrite.newLine();
			}
			
			else if (dvalue <= 0.9 && dvalue>0.1) {
				trainwrite.write(info);
				trainwrite.newLine();
			} else {
				testwrite.write(info);
				testwrite.newLine();
				if(arr.length==2)
				{
					testmetawrite.write(arr[1] + " " + testline);
					testmetawrite.newLine();
					testline++;
				}
			}
		}
		
		br.close();
		fr.close();

		trainwrite.flush();
		trainwrite.close();
		
		testwrite.flush();
		testwrite.close();

		validwrite.flush();
		validwrite.close();
		
		testmetawrite.flush();
		testmetawrite.close();
	}
}