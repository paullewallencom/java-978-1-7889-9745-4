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

import java.io.*;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/*
 * @author: Md. Rezaul Karim. However, this an extended version of original example provided in GitHub
 */

public class MovieLensFormaterWithMetaData {
	private static String inputfilepath;
	private static String outputfilepath;
	private static int targetcolumn = 0;
	private static String deletecolumns = "3";
	private static String separator = "::";
	private static int offset = 0;

	public static void main(String[] args) throws Exception {
		String foldername = "ml-1m";
		String outFolder = "formatted_data";

		Set<Integer> deletecolumnsset = new HashSet<Integer>();
		Map<String, Integer> valueidmap = new HashMap<String, Integer>();
		
		targetcolumn = 2; // movielens format
		String[] deletecolumnarr = deletecolumns.split(";");
		
		for (String deletecolumn : deletecolumnarr) {
			deletecolumnsset.add(Integer.parseInt(deletecolumn));
		}

		inputfilepath = foldername + File.separator + "users.dat";
		Reader fr = new FileReader(inputfilepath);
		BufferedReader br = new BufferedReader(fr);
		
		Map<Integer, String> usergenemap = new HashMap<Integer, String>();
		Map<Integer, String> useragemap = new HashMap<Integer, String>();
		Map<Integer, String> useroccupationmap = new HashMap<Integer, String>();
		
		String line;
		while (br.ready()) {
			line = br.readLine();
			String[] arr = line.split(separator);
			usergenemap.put(Integer.parseInt(arr[0]), arr[1]);
			useragemap.put(Integer.parseInt(arr[0]), arr[2]);
			useroccupationmap.put(Integer.parseInt(arr[0]), arr[3]);
		}
		
		br.close();
		fr.close();

		inputfilepath = foldername + File.separator + "movies.dat";
		fr = new FileReader(inputfilepath);
		br = new BufferedReader(fr);
		Map<Integer, String> moviemap = new HashMap<Integer, String>();
		
		while (br.ready()) {
			line = br.readLine();
			String[] arr = line.split(separator);
			moviemap.put(Integer.parseInt(arr[0]), arr[2]);
		}
		
		br.close();
		fr.close();

		inputfilepath = foldername + File.separator + "ratings.dat";
		outputfilepath = outFolder + File.separator + "ratings.libfm";
		BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(outputfilepath)));

		fr = new FileReader(inputfilepath);
		br = new BufferedReader(fr);

		while (br.ready()) {
			line = br.readLine();
			String[] arr = line.split(separator);
			StringBuilder sb = new StringBuilder();
			sb.append(arr[targetcolumn]);
			
			int columnidx = 0;
			int userid = Integer.parseInt(arr[0]);
			int movieid = Integer.parseInt(arr[1]);
			
			for (int i = 0; i < arr.length; i++) {
				if (i != targetcolumn && !deletecolumnsset.contains(i)) {
					String useroritemid = Integer.toString(columnidx) + " " + arr[i];
					
					if (!valueidmap.containsKey(useroritemid)) {
						valueidmap.put(useroritemid, offset++);
					}
					
					sb.append(" ");
					sb.append(valueidmap.get(useroritemid));
					sb.append(":1");

					columnidx++;
				}
			}
			
			// Add attributes
			String gender = usergenemap.get(userid);
			String attributeid = "The gender information " + gender;
			
			if (!valueidmap.containsKey(attributeid)) {
				valueidmap.put(attributeid, offset++);
			}

			sb.append(" ");
			sb.append(valueidmap.get(attributeid));
			sb.append(":1");

			String age = useragemap.get(userid);
			attributeid = "The age information " + age;
			
			if (!valueidmap.containsKey(attributeid)) {
				valueidmap.put(attributeid, offset++);
			}

			sb.append(" ");
			sb.append(valueidmap.get(attributeid));
			sb.append(":1");

			String occupation = useroccupationmap.get(userid);
			attributeid = "The occupation information " + occupation;
			
			if (!valueidmap.containsKey(attributeid)) {
				valueidmap.put(attributeid, offset++);
			}

			sb.append(" ");
			sb.append(valueidmap.get(attributeid));
			sb.append(":1");

			String movieclassdesc = moviemap.get(movieid);
			String[] movieclassarr = movieclassdesc.split("\\|");
			
			for (String movieclass : movieclassarr) {
				attributeid = "The movie class information " + movieclass;
				if (!valueidmap.containsKey(attributeid)) {
					valueidmap.put(attributeid, offset++);
				}

				sb.append(" ");
				sb.append(valueidmap.get(attributeid));
				sb.append(":1");
			}
			
			//add metadata information, userid and movieid
			sb.append("#");
			sb.append(userid);
			sb.append(" "+movieid);
			writer.write(sb.toString());
			writer.newLine();
		}
		
		br.close();
		fr.close();

		writer.flush();
		writer.close();
	}
}
