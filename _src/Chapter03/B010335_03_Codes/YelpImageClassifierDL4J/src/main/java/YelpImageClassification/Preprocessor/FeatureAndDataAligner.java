package YelpImageClassification.Preprocessor;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

import org.apache.commons.lang3.tuple.ImmutableTriple;
import org.apache.commons.lang3.tuple.Triple;

// initializing alignedData with empty labMap when it is not provided (we are working with training data)
public class FeatureAndDataAligner {

    private Map<Integer, List<Integer>> dataMap;
    private Map<Integer, String> bizMap;
    private Optional<Map<String, Set<Integer>>> labMap;
    private List<Integer> rowindices;

    public FeatureAndDataAligner(Map<Integer, List<Integer>> dataMap, Map<Integer, String> bizMap, Optional<Map<String, Set<Integer>>> labMap) {
        this(dataMap, bizMap, labMap, dataMap.keySet().stream().collect(Collectors.toList()));
    }

    public FeatureAndDataAligner(Map<Integer, List<Integer>> dataMap, Map<Integer, String> bizMap) {
        this(dataMap, bizMap, Optional.empty(), dataMap.keySet().stream().collect(Collectors.toList()));
    }

    public FeatureAndDataAligner(Map<Integer, List<Integer>> dataMap,Map<Integer, String> bizMap, Optional<Map<String, Set<Integer>>> labMap,List<Integer> rowindices) {
        this.dataMap = dataMap;
        this.bizMap = bizMap;
        this.labMap = labMap;
        this.rowindices = rowindices;
    }

    public List<Triple<Integer, String, List<Integer>>> alignBusinessImgageIds(Map<Integer, List<Integer>> dataMap, Map<Integer, String> bizMap) {
        return alignBusinessImgageIds(dataMap, bizMap, dataMap.keySet().stream().collect(Collectors.toList()));
    }

    public List<Triple<Integer, String, List<Integer>>> alignBusinessImgageIds(Map<Integer, List<Integer>> dataMap, Map<Integer, String> bizMap,List<Integer> rowindices) {
        ArrayList<Triple<Integer, String, List<Integer>>> result = new ArrayList<>();
        for (Integer pid : rowindices) {
            Optional<String> imgHasBiz = Optional.ofNullable(bizMap.get(pid));
            String bid = imgHasBiz.orElse("-1");
            if (dataMap.containsKey(pid) && imgHasBiz.isPresent()) {
               result.add(new ImmutableTriple<>(pid, bid, dataMap.get(pid)));
            }
        }
        return result;
    }

    private List<Quarta<Integer, String, List<Integer>, Set<Integer>>> alignLabels(Map<Integer, List<Integer>> dataMap,
                                                                                   Map<Integer, String> bizMap,
                                                                                   Optional<Map<String, Set<Integer>>> labMap,
                                                                                   List<Integer> rowindices) {
        ArrayList<Quarta<Integer, String, List<Integer>, Set<Integer>>> result = new ArrayList<>();
        List<Triple<Integer, String, List<Integer>>> a1 = alignBusinessImgageIds(dataMap, bizMap, rowindices);
        for (Triple<Integer, String, List<Integer>> p : a1) {
            String bid = p.getMiddle();
            Set<Integer> labs = Collections.emptySet();
            if (labMap.isPresent() && labMap.get().containsKey(bid)) {
                 labs = labMap.get().get(bid);
            }
            result.add(new Quarta<>(p.getLeft(), p.getMiddle(), p.getRight(), labs));
        }
        return result;
    }

    private volatile List<Quarta<Integer, String, List<Integer>, Set<Integer>>> _data = null;

    // pre-computing and saving data as a val so method does not need to re-compute each time it is called.
    public List<Quarta<Integer, String, List<Integer>, Set<Integer>>> data() {
        if (_data == null) {
            synchronized (this) {
                if (_data == null) {
                    _data = alignLabels(dataMap, bizMap, labMap, rowindices);
                }
            }
        }
        return _data;
    }

    // getter functions
    public List<Integer> getImgIds() {
        return data().stream().map(e -> e.a).collect(Collectors.toList());
    }
    public List<String> getBusinessIds() {
        return data().stream().map(e -> e.b).collect(Collectors.toList());
    }
    public List<List<Integer>> getImgVectors() {
        return data().stream().map(e -> e.c).collect(Collectors.toList());
    }
    public List<Set<Integer>> getBusinessLabels() {
        return data().stream().map(e -> e.d).collect(Collectors.toList());
    }
    public Map<String, Integer> getImgCntsPerBusiness() {
        return getBusinessIds().stream().collect(Collectors.groupingBy(Function.identity())).entrySet()
                .stream().map(e -> new AbstractMap.SimpleEntry<>(e.getKey(), e.getValue().size()))
                .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()));
    }

    public static class Quarta <A, B, C, D> {
        public final A a;
        public final B b;
        public final C c;
        public final D d;

        public Quarta(A a, B b, C c, D d) {
            this.a = a;
            this.b = b;
            this.c = c;
            this.d = d;
        }
    }
}
