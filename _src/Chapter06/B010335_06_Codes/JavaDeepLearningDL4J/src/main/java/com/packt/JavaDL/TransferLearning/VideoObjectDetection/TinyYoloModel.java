package com.packt.JavaDL.TransferLearning.VideoObjectDetection;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.objdetect.DetectedObject;
import org.deeplearning4j.nn.layers.objdetect.Yolo2OutputLayer;
import org.deeplearning4j.zoo.model.TinyYOLO;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_imgproc.putText;
import static org.bytedeco.javacpp.opencv_imgproc.rectangle;


public class TinyYoloModel {

    private static ComputationGraph model;
    private List<DetectedObject> predictedObjects;
    private HashMap<Integer, String> labels;

    private TinyYoloModel() {
        try {
            model = (ComputationGraph) new TinyYOLO().initPretrained();
            createObjectLabels();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static final TinyYoloModel yolo = new TinyYoloModel();   

    public static TinyYoloModel getPretrainedModel() {
        return yolo;
    }
    
    public static String getSummary() {
        return model.summary();
    }

    public void markObjectWithBoundingBox(Mat file, int imageWidth, int imageHeight, boolean newBoundingBOx,String winName) throws Exception {
        int W = 640; // width of the video frame 
        int H = 360; // Height of the video frame
        int gW = 13; // Grid width
        int gH = 13; // Grid Height
        double dT = 0.5; // Detection threshold

        Yolo2OutputLayer outputLayer = (Yolo2OutputLayer) model.getOutputLayer(0);
        if (newBoundingBOx) {
            INDArray indArray = prepareImage(file, W, H);
            INDArray results = model.outputSingle(indArray);
            predictedObjects = outputLayer.getPredictedObjects(results, dT);
            System.out.println("results = " + predictedObjects);
            markObjectWithBoundingBox(file, gW, gH, imageWidth, imageHeight);
        } else {
        	markObjectWithBoundingBox(file, gW, gH, imageWidth, imageHeight);
        }
        imshow(winName, file);
    }

    INDArray prepareImage(Mat file, int width, int height) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(height, width, 3);
        ImagePreProcessingScaler imagePreProcessingScaler = new ImagePreProcessingScaler(0, 1);
        INDArray indArray = loader.asMatrix(file);
        imagePreProcessingScaler.transform(indArray);
        return indArray;
    }

    void createObjectLabels() {
        if (labels == null) {
            String label = "aeroplane\n" + "bicycle\n" + "bird\n" + "boat\n" + "bottle\n" + "bus\n" + "car\n" +
                    "cat\n" + "chair\n" + "cow\n" + "diningtable\n" + "dog\n" + "horse\n" + "motorbike\n" +
                    "person\n" + "pottedplant\n" + "sheep\n" + "sofa\n" + "train\n" + "tvmonitor";
            String[] split = label.split("\\n");
            int i = 0;
            labels = new HashMap<>();
            for (String label1 : split) {
                labels.put(i++, label1);
            }
        }
    }

    void markObjectWithBoundingBox(Mat file, int gridWidth, int gridHeight, int w, int h) {

        if (predictedObjects == null) {
            return;
        }
        ArrayList<DetectedObject> detectedObjects = new ArrayList<>(predictedObjects);

        while (!detectedObjects.isEmpty()) {
            Optional<DetectedObject> max = detectedObjects.stream().max((o1, o2) -> ((Double) o1.getConfidence()).compareTo(o2.getConfidence()));
            if (max.isPresent()) {
                DetectedObject maxObjectDetect = max.get();
                removeObjectsIntersectingWithMax(detectedObjects, maxObjectDetect);
                detectedObjects.remove(maxObjectDetect);
                markObjectWithBoundingBox(file, gridWidth, gridHeight, w, h, maxObjectDetect);
            }
        }
    }

    static void removeObjectsIntersectingWithMax(ArrayList<DetectedObject> detectedObjects, DetectedObject maxObjectDetect) {
        double[] bottomRightXY1 = maxObjectDetect.getBottomRightXY();
        double[] topLeftXY1 = maxObjectDetect.getTopLeftXY();
        List<DetectedObject> removeIntersectingObjects = new ArrayList<>();
        for (DetectedObject detectedObject : detectedObjects) {
            double[] topLeftXY = detectedObject.getTopLeftXY();
            double[] bottomRightXY = detectedObject.getBottomRightXY();
            double iox1 = Math.max(topLeftXY[0], topLeftXY1[0]);
            double ioy1 = Math.max(topLeftXY[1], topLeftXY1[1]);

            double iox2 = Math.min(bottomRightXY[0], bottomRightXY1[0]);
            double ioy2 = Math.min(bottomRightXY[1], bottomRightXY1[1]);

            double inter_area = (ioy2 - ioy1) * (iox2 - iox1);

            double box1_area = (bottomRightXY1[1] - topLeftXY1[1]) * (bottomRightXY1[0] - topLeftXY1[0]);
            double box2_area = (bottomRightXY[1] - topLeftXY[1]) * (bottomRightXY[0] - topLeftXY[0]);

            double union_area = box1_area + box2_area - inter_area;
            double iou = inter_area / union_area;


            if (iou > 0.5) {
                removeIntersectingObjects.add(detectedObject);
            }

        }
        detectedObjects.removeAll(removeIntersectingObjects);
    }

    void markObjectWithBoundingBox(Mat file, int gridWidth, int gridHeight, int w, int h, DetectedObject obj) {

        double[] xy1 = obj.getTopLeftXY();
        double[] xy2 = obj.getBottomRightXY();
        int predictedClass = obj.getPredictedClass();
        int x1 = (int) Math.round(w * xy1[0] / gridWidth);
        int y1 = (int) Math.round(h * xy1[1] / gridHeight);
        int x2 = (int) Math.round(w * xy2[0] / gridWidth);
        int y2 = (int) Math.round(h * xy2[1] / gridHeight);
        rectangle(file, new Point(x1, y1), new Point(x2, y2), Scalar.RED);
        putText(file, labels.get(predictedClass), new Point(x1 + 2, y2 - 2), FONT_HERSHEY_DUPLEX, 1, Scalar.GREEN);
    }

}