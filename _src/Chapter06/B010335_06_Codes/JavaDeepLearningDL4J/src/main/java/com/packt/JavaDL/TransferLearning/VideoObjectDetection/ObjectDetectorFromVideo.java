package com.packt.JavaDL.TransferLearning.VideoObjectDetection;

import static org.bytedeco.javacpp.opencv_highgui.destroyAllWindows;
import static org.bytedeco.javacpp.opencv_highgui.imshow;
import static org.bytedeco.javacpp.opencv_highgui.waitKey;
import java.io.IOException;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameConverter;

public class ObjectDetectorFromVideo {
    private volatile Mat[] v = new Mat[1];
    private String windowName;

    public static void main(String[] args) throws java.lang.Exception {
        String videoPath = "data/videoSample.mp4";
        TinyYoloModel model = TinyYoloModel.getPretrainedModel();
        
        System.out.println(TinyYoloModel.getSummary());
        new ObjectDetectorFromVideo().startRealTimeVideoDetection(videoPath, model);
    }

    public void startRealTimeVideoDetection(String videoFileName, TinyYoloModel model) throws java.lang.Exception {
        windowName = "Object Detection from Video";
        FFmpegFrameGrabber frameGrabber = new FFmpegFrameGrabber(videoFileName);
        frameGrabber.start();

        Frame frame;
        double frameRate = frameGrabber.getFrameRate();
        System.out.println("The inputted video clip has " + frameGrabber.getLengthInFrames() + " frames");
        System.out.println("The inputted video clip has frame rate of " + frameRate);

        try {
            for(int i = 1; i < frameGrabber.getLengthInFrames(); i+=(int)frameRate) {
                frameGrabber.setFrameNumber(i);
                frame = frameGrabber.grab();
                v[0] = new OpenCVFrameConverter.ToMat().convert(frame);
                model.markObjectWithBoundingBox(v[0], frame.imageWidth, frame.imageHeight, true, windowName);
                imshow(windowName, v[0]);

                char key = (char) waitKey(20);
                // Exit on escape:
                if (key == 27) {
                    destroyAllWindows();
                    break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            frameGrabber.stop();
        }
        frameGrabber.close();
    }
}