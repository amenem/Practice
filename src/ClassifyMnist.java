import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.ContinuousValueEncoder;

public class ClassifyMnist {
	public static void main(String args[]) throws IOException {
		// read the file

		// OnlineLogisticRegression lr=new OnlineLogisticRegression(10,784,new
		// L2());
		CrossFoldLearner cf = new CrossFoldLearner(5, 10, 784, new L1());
		// cf.addModel(lr);

		String line;
		List<ContinuousValueEncoder> encoders = new LinkedList<ContinuousValueEncoder>();
		for (int i = 1; i < 785; i++) {
			encoders.add(new ContinuousValueEncoder("feature_" + i));
		}
		double value;
		// for(int loop=0;loop<1;loop++){
		BufferedReader br = new BufferedReader(new FileReader(new File(
				"data/train.csv")));
		br.readLine();// ignore header
		while ((line = br.readLine()) != null) {
			// for each line we have 784 features list column for the tables in
			// integer
			String[] inputArr = line.split(",");
			Vector v = new RandomAccessSparseVector(784);
			int targetValue = Integer.parseInt(inputArr[0]);

			for (int i = 1; i < 785; i++) {
				value = Double.parseDouble(inputArr[i]);
				encoders.get(i - 1).addToVector((byte[]) null, value, v);

			}
			// System.out.println(targetValue+" "+v);
			cf.train(targetValue, v);

			// }
		}
		System.out.println(cf.percentCorrect());

		br.close();
		cf.close();

	}

}
