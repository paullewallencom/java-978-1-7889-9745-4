package YelpImageClassification.Preprocessor;

import java.io.*;
import java.util.*;
import java.util.function.BinaryOperator;
import java.util.stream.Collectors;

public class CSVImageMetadataReader {
    private static List<Integer> DEFAULT_ROWS = Arrays.asList(-1);

    public static List<List<String>> readMetadata(String csv) throws IOException {
        return readMetadata(csv, DEFAULT_ROWS);
    }

    public static List<List<String>> readMetadata(String csv, List<Integer> rows) throws IOException {
        boolean defaultRows = rows.size() == 1 && rows.get(0) == -1;
        LinkedList<Integer> rowsCopy = null;
        if (!defaultRows) {
            rowsCopy = new LinkedList<>(rows);
        }
        try(BufferedReader bufferedReader =
                    new BufferedReader(new InputStreamReader(new FileInputStream(new File(csv))))) {
            ArrayList<List<String>> arrayList = new ArrayList<>();
            String line = bufferedReader.readLine();
            int i = 0;
            while (line != null) {
                if (defaultRows || rowsCopy.getFirst() == i) {
                    if (!defaultRows) {
                        rowsCopy.removeFirst();
                    }
                    arrayList.add(Arrays.asList(line.split(",")));
                }
                line = bufferedReader.readLine();
                i++;
            }
            return arrayList;
        }
    }

    public static Map<String, Set<Integer>> readBusinessLabels(String csv) throws IOException {
        return readBusinessLabels(csv, DEFAULT_ROWS);
    }

    public static Map<String, Set<Integer>> readBusinessLabels(String csv, List<Integer> rows) throws IOException {
        return readMetadata(csv, rows).stream()
                .skip(1)
                .map(l -> parseBusinessLabelsKv(l))
                .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()));
    }

    public static Map<Integer, String> readBusinessToImageLabels(String csv) throws IOException {
        return readBusinessToImageLabels(csv, DEFAULT_ROWS);
    }

    public static Map<Integer, String> readBusinessToImageLabels(String csv, List<Integer> rows) throws IOException {
        return readMetadata(csv, rows).stream()
                .skip(1)
                .map(l -> parseBusinessToImageLabelsKv(l))
                .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue(), useLastMerger()));
    }

    private static AbstractMap.SimpleEntry<String, Set<Integer>> parseBusinessLabelsKv(List<String> list) {
        if (list.size() == 1) {
            return new AbstractMap.SimpleEntry<String, Set<Integer>>(list.get(0).toString(), Collections.emptySet());
        } else {
            Set<Integer> value = Arrays.asList(list.get(1).split(" "))
                    .stream()
                    .map((s) -> Integer.parseInt(s))
                    .collect(Collectors.toSet());
            return new AbstractMap.SimpleEntry<String, Set<Integer>>(list.get(0).toString(), value);
        }
    }

    private static AbstractMap.SimpleEntry<Integer,String> parseBusinessToImageLabelsKv(List<String> list) {
        if (list.size() == 1) {
            return new AbstractMap.SimpleEntry<Integer, String>(Integer.parseInt(list.get(0)), "-1");
        } else {
            return new AbstractMap.SimpleEntry<Integer, String>(Integer.parseInt(list.get(0)), list.get(1).split(" ")[0]);
        }
    }

    private static <T> BinaryOperator<T> useLastMerger() {
        return (u,v) -> v;
    }
}