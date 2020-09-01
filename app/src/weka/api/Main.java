package weka.api;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Scanner;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/*
 * Machine Learning - 3. domaci, 2. zadatak
 * @author Nikola Pujaz 21m/19
 *  
 * */

public class Main {
	public static void main(String[] args) throws Exception {
		
		DatasetManager dataMng = new DatasetManager();
		ClassificationManager clsMng = new ClassificationManager();
		
		DataSource dataSource = new DataSource("data/nursery.arff");
		
		dataMng.setPercentageSplitSet(dataSource.getDataSet(), 66);
		
		Instances trainSet = dataMng.getTrainSet();
		Instances testSet = dataMng.getTestSet();

		clsMng.svmTrainTest(trainSet, testSet);
		clsMng.j48TrainTest(trainSet, testSet);
		
		System.out.println("Please put the file you want to classify into the data folder and type in the name of the file: ");
		Scanner scanner = new Scanner(System.in);
		String fileName = scanner.nextLine();
		System.out.println("Now select classifier by typing J48 or SVM:");
		String classifier = scanner.nextLine();
		scanner.close();
		
		Instances labeled = null;
		switch (classifier) {
		case "SVM":
			labeled = clsMng.classifyInstances(fileName, "SVM");
			break;
		case "J48":
			labeled = clsMng.classifyInstances(fileName, "J48");
			break;
		default:
			System.out.println("Entered command is not recognized!");
			break;
		}
		
		//Writing labeled instances into a file
		BufferedWriter fileWriter = new BufferedWriter(new FileWriter("data/labeled-" + fileName + ".arff"));
		fileWriter.write(labeled.toString());
		fileWriter.newLine();
		fileWriter.flush();
		fileWriter.close();
		
		System.out.println("Labeled instances successfully and placed them into a file: labeled-" + fileName + ".arff");
	}

}
