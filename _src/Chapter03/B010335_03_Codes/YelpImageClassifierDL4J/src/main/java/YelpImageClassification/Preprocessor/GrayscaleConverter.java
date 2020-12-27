package YelpImageClassification.Preprocessor;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;


public class GrayscaleConverter {
    public static void main(String[] args) throws IOException {
        BufferedImage testImage = ImageIO.read(new File("data/images/preprocessed/147square.jpg"));
        BufferedImage grayImage = makeGray(testImage);
        ImageIO.write(grayImage, "jpg", new File("data/images/preprocessed/147gray.jpg"));
    }

    private static int pixels2Gray(int R, int G, int B) {
        return (R + G + B) / 3;
    }

    private static BufferedImage makeGray(BufferedImage testImage) {
        int w = testImage.getWidth();
        int h = testImage.getHeight();
        for (int w1 = 0; w1 < w; w1++) {
            for (int h1 = 0; h1 < h; h1++) {
                int col = testImage.getRGB(w1, h1);
                int R = (col & 0xff0000) / 65536;
                int G = (col & 0xff00) / 256;
                int B = (col & 0xff);
                int graycol = pixels2Gray(R, G, B);
                testImage.setRGB(w1, h1, new Color(graycol, graycol, graycol).getRGB());
            }
        }
        return testImage;
    }
}
