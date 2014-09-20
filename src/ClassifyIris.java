 
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

import java.util.List;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.Dictionary;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.L1;
 
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;

import com.google.common.collect.Lists;

public class ClassifyIris {

	public static void main(String args[]) throws IOException {
		// read the input file in BufferedReader
		BufferedReader br = new BufferedReader(new FileReader(new File(
				"data/iris.csv")));
		int numFeatures = br.readLine().split(",").length - 1;
		OnlineLogisticRegression lr = new OnlineLogisticRegression(3, 4,
				new L1());
		String line;
		List<Vector> data = Lists.newArrayList();
		List<Integer> target = Lists.newArrayList();
		Dictionary dict = new Dictionary();
		while ((line = br.readLine()) != null) {
			Vector v = new DenseVector(numFeatures);
			String input[] = line.split(",");
			for (int i = 0; i < input.length; i++) {

				if (i == input.length - 1) {
					int targetValue = dict.intern(input[i]);// target value
					// encoding
					target.add(targetValue);
				} else {
					v.set(i, Double.parseDouble(input[i]));
				}
			}
			data.add(v);
		}

		CrossFoldLearner cf = new CrossFoldLearner(6, 3, 4, new L1());
		cf.addModel(lr);
		for (int loop = 0; loop < 30; loop++) //sign of overfit ??
			for (int j = 0; j < data.size(); j++) {
				//lr.train(target.get(j), data.get(j));
				cf.train(target.get(j), data.get(j));
			}
		int l = 0;
		int n = 0;
		for (Vector v : data) {
			l += cf.classifyFull(v).maxValueIndex() == target.get(n++) ? 1 : 0;
			// System.out.println(v+" "+cf.classifyFull(v).maxValueIndex()+" "+target.get(n++));
		}
		System.out.println(l * 100.0 / n);
		cf.close();
		br.close();

	}
}
