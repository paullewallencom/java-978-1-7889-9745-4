package YelpImageClassification.Trainer;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NetworkSaver {
	public static MultiLayerNetwork loadNN(String NNconfig, String NNparams) throws IOException {
		// get neural network config
		@SuppressWarnings("deprecation")
		MultiLayerConfiguration confFromJson = MultiLayerConfiguration
				.fromJson(FileUtils.readFileToString(new File(NNconfig)));

		// get neural network parameters
		DataInputStream dis = new DataInputStream(new FileInputStream(NNparams));
		INDArray newParams = Nd4j.read(dis);

		// creating network object
		MultiLayerNetwork savedNetwork = new MultiLayerNetwork(confFromJson);
		savedNetwork.init();
		savedNetwork.setParameters(newParams);

		return savedNetwork;
	}

	@SuppressWarnings("deprecation")
	public void saveNN(MultiLayerNetwork model, String NNconfig, String NNparams) throws IOException {
		// save neural network config
		FileUtils.write(new File(NNconfig), model.getLayerWiseConfigurations().toJson());

		// save neural network parms
		DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(NNparams)));
		Nd4j.write(model.params(), dos);
	}
}
