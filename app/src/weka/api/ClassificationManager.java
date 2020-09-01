package weka.api;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;

/*
 * 2. SVM
 * 3. J48
 * 4. Method for Instance Classification
 * @author Nikola Pujaz 21m/19
 * */

public class ClassificationManager {
	
	public ClassificationManager() {
		super();
		// TODO Auto-generated constructor stub
	}

	public void svmTrainTest(Instances trainSet, Instances testSet) throws Exception {
		
		//Default options
		SMO svm = new SMO();		
		svm.buildClassifier(trainSet);
		
		// Save Model
		SerializationHelper.write("models/svm-nursery.model", svm);
		
		// 10-fold cross-validation
		Evaluation evaluationCV = new Evaluation(trainSet);
		evaluationCV.crossValidateModel(svm, trainSet, 10, new Random(1));
		
		System.out.println(evaluationCV.toSummaryString("SVM (10-fold cross-validation) Results:", false));
		
		// Test Set Evaluation
		Evaluation evaluationTest = new Evaluation(trainSet);
		evaluationTest.evaluateModel(svm, testSet);
		System.out.println(evaluationTest.toSummaryString("SVM (Test Set Evaluation) Results:", false));
		
		writeResultsToFile(evaluationCV, evaluationTest, "SVM (10-fold cross-validation) Results:","SVM (Test Set Evaluation) Results:");
	}
	
	public void j48TrainTest(Instances trainSet, Instances testSet) throws Exception {
		
		// Default options
		J48 j48 = new J48();
		j48.buildClassifier(trainSet);
		
		// Save Model
		SerializationHelper.write("models/j48-nursery.model", j48);
	
		// 10-fold cross-validation
		Evaluation evaluationCV = new Evaluation(trainSet);
		evaluationCV.crossValidateModel(j48, trainSet, 10, new Random(1));
		
		// Test Set Evaluation
		Evaluation evaluationTest = new Evaluation(trainSet);
		evaluationTest.evaluateModel(j48, testSet);
		
		// J48 Pruned tree
		System.out.println(j48);
		
		// Graph that can be visualised with UI libs
		System.out.println(j48.graph());
		
		writeResultsToFile(evaluationCV, evaluationTest, "J48 (10-fold cross-validation) Results:", "J48 (Test Set Evaluation) Results:");
	}
	
	public Instances classifyInstances(String fileName, String classifier) throws Exception {
		
		Instances unlabeled = new Instances(new BufferedReader(new FileReader("data/" + fileName + ".arff")));
		unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
		
		Instances labeled = new Instances(unlabeled);
		
		if(classifier.equals("svm")) {
			SMO svm = (SMO) SerializationHelper.read("models/svm-nursery.model");
			for(int i = 0; i < unlabeled.numInstances(); i++) {
				double clsLabel = svm.classifyInstance(unlabeled.instance(i));
				labeled.instance(i).setClassValue(clsLabel);
			}
		} else {
			J48 j48 = (J48)	SerializationHelper.read("models/j48-nursery.model");
			for(int i = 0; i < unlabeled.numInstances(); i++) {
				double clsLabel = j48.classifyInstance(unlabeled.instance(i));
				labeled.instance(i).setClassValue(clsLabel);
			}
		}
		
		return labeled;
	}
	
	private static void writeResultsToFile(Evaluation evaluationCV, Evaluation evaluationTest, String classifierCV, String classifierTest) throws Exception {			
		
		BufferedWriter fileWriter = new BufferedWriter(new FileWriter("results/results.txt", true));
		fileWriter.write(evaluationCV.toSummaryString(classifierCV, false));
		fileWriter.write(evaluationCV.toMatrixString(">>>> Overall Confusion Matrix <<<<"));
		fileWriter.write("======================\n");
		fileWriter.write(evaluationTest.toSummaryString(classifierTest, false));
		fileWriter.write(evaluationTest.toMatrixString(">>>> Overall Confusion Matrix <<<<"));
		fileWriter.write("======================\n");
		fileWriter.flush();
		fileWriter.close();
	}
	
}
