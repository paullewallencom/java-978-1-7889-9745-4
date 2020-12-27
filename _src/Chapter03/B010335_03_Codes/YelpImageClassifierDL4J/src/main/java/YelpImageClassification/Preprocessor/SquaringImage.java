package YelpImageClassification.Preprocessor;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.imgscalr.Scalr;

public class SquaringImage {
    public static void main(String[] args) throws IOException {
        BufferedImage myimg = ImageIO.read(new File("data/images/train/147.jpg"));
        BufferedImage myimgSquare = makeSquare(myimg);
        ImageIO.write(myimgSquare, "jpg", new File("data/images/preprocessed/147square.jpg"));
    }

    private static BufferedImage makeSquare(BufferedImage img) {
        int w = img.getWidth();
        int h = img.getHeight();
        int dim = Math.min(w, h);

        if (w == h) {
            return img;
        } else if (w > h) {
            return Scalr.crop(img, (w - h) / 2, 0, dim, dim);
        } else {
            return Scalr.crop(img, 0, (h - w) / 2, dim, dim);
        }
    }
}
