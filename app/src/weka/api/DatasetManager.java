package weka.api;

import java.util.Random;

import weka.core.Instances;

/*
 * 1. Percentage Split 66/34%
 * @author Nikola Pujaz 21m/19
 * */

public class DatasetManager {
	
	private Instances trainSet = null;
	
	private Instances testSet = null;
	
	public DatasetManager() {
		super();
	}
	
	public DatasetManager(Instances trainSet, Instances testSet) {
		super();
		this.trainSet = trainSet;
		this.testSet = testSet;
	}

	public Instances getTrainSet() {
		return trainSet;
	}

	public void setTrainSet(Instances trainSet) {
		this.trainSet = trainSet;
	}

	public Instances getTestSet() {
		return testSet;
	}

	public void setTestSet(Instances testSet) {
		this.testSet = testSet;
	}

	private Instances randomizeSet(Instances dataset) {
		dataset.randomize(new Random(42));
		return dataset;
	}
	
	public void setPercentageSplitSet(Instances dataset, int percentage) throws Exception {
		
		// Setting the subset sizes for the split
		int trainSetSize = Math.round((dataset.numInstances() * percentage)/100);
		int testSetSize = dataset.numInstances() - trainSetSize;
		
		dataset = randomizeSet(dataset);
		
		trainSet = new Instances(dataset, 0, trainSetSize);
		testSet = new Instances(dataset, trainSetSize, testSetSize);
		
		// In case there is no class attribute defined
		trainSet.setClassIndex(trainSet.numAttributes() - 1);
		testSet.setClassIndex(testSet.numAttributes() - 1);
	}
}
