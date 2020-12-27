package YelpImageClassification.Preprocessor;

import static YelpImageClassification.Preprocessor.imageUtils.image2gray;
import static YelpImageClassification.Preprocessor.imageUtils.makeSquare;
import static YelpImageClassification.Preprocessor.imageUtils.resizeImg;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import javax.imageio.ImageIO;

public class imageFeatureExtractor {
    /**  Define RegEx to extract jpg name from the image class which is used to match against training labels */
    public static Pattern patt_get_jpg_name = Pattern.compile("[0-9]");

    /** Collects all images associated with a BizId. */
    public static List<Integer> getImgIdsFromBusinessId(Map<Integer, String> bizMap, List<String> businessIds) {
        return bizMap.entrySet().stream().filter(x -> businessIds.contains(x.getValue())).map(Map.Entry::getKey)
                .collect(Collectors.toList());
    }

    /** Get a list of images to load and process
     *
     * @param photoDir directory where the raw images reside
     * @param ids optional parameter to subset the images loaded from photoDir.
     */
    public static List<String> getImageIds(String photoDir, Map<Integer, String> businessMap, List<String> businessIds) {
        File d = new File(photoDir);
        List<String> imgsPath = Arrays.stream(d.listFiles()).map(f -> f.toString()).collect(Collectors.toList());

        boolean defaultBusinessMap = businessMap.size() == 1 && businessMap.get(-1).equals("-1");
        boolean defaultBusinessIds = businessIds.size() == 1 && businessIds.get(0).equals("-1");
        if (defaultBusinessMap || defaultBusinessIds) {
            return imgsPath;
        } else {
            Map<Integer, String> imgsMap = imgsPath.stream()
                    .map(x -> new AbstractMap.SimpleEntry<Integer, String>(extractInteger(x), x))
                    .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()));
            List<Integer> imgsPathSub = imageFeatureExtractor.getImgIdsFromBusinessId(businessMap, businessIds);
            return imgsPathSub.stream().filter(x -> imgsMap.containsKey(x)).map(x -> imgsMap.get(x))
                    .collect(Collectors.toList());
        }
    }

    /** Read and process images into a photoID -> vector map
     *
     * @param imgs list of images to read-in.  created from getImageIds function.
     * @param resizeImgDim dimension to rescale square images to
     * @param nPixels number of pixels to maintain.  mainly used to sample image to drastically reduce runtime while testing features.
     *
     */
    public static Map<Integer, List<Integer>> processImages(List<String> imgs, int resizeImgDim, int nPixels) {
        Function<String, AbstractMap.Entry<Integer, List<Integer>>> handleImg = x -> {
            BufferedImage img = null;
            try {
                img = ImageIO.read(new File(x));
            } catch (IOException e) {
                e.printStackTrace();
            }
            img = makeSquare(img);
            img = resizeImg(img, resizeImgDim, resizeImgDim);
            List<Integer> value = image2gray(img);
            if(nPixels != -1) {
                value = value.subList(0, nPixels);
            }
            return new AbstractMap.SimpleEntry<Integer, List<Integer>>(extractInteger(x), value);
        };

        return imgs.stream().map(handleImg).filter(e -> !e.getValue().isEmpty())
                .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()));
    }

    public static Map<Integer, List<Integer>> processImages(List<String> imgs, int resizeImgDim) {
        return processImages(imgs, resizeImgDim, -1);
    }

    public static Map<Integer, List<Integer>> processImages(List<String> imgs) {
        return processImages(imgs, 128);
    }

    private static Integer extractInteger(String path) {
        StringBuilder sb = new StringBuilder();
        Matcher m = patt_get_jpg_name.matcher(path);
        while (m.find()) {
            sb.append(m.group());
        }
        return Integer.parseInt(sb.toString());
    }
}
