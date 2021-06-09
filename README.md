# Structuring Machine Learning Projects
## Introduction to ML Strategy: Orthogonalization
* Different functions are performed by different devices so that the functionally becomes discrete and is not mixed up
* For Machine Learning Systems to work well we need 4 things to focus on:
  * Fit training set well on cost function
    * If system is not working properly in this area then we can use
      * Training a bigger network
      * Train a better optimization algorithm like Adam optimization algo
  * Fit dev set on cost function
    * If system is not working properly in this area then we can use
      * Regularization
      * Bigger training set
  * Fit test set well on cost function
    * If system is not working properly in this area then we can use
      * Get a bigger dev set
  * Perform well in real world
    * If system is not working in this area we can use
    * Change the dev set or the cost function
## Setting up your goals
### Single number evaluation metric
* Precision: of the examples recognised as cats, how much % are actually cats
* Recall: what % of actually cats are correctly recognised

Classifier | Precision | Recall
:-- | :--: | :--: 
A | 95% | 90%
B | 98% | 85%

* In the above table, classifier A has better recall but Classifier B has better precision, so which classifier to use?
* Create a new scoring matrix F1 Score

Classifier | Precision | Recall | F1 Score
:-- | :--: | :--: | :--:
A | 95% | 90% | 92.4%
B | 98% | 85% | 91.0%

F1 Score(Harmonic Mean) = 2/((1/P) + (1/R))

### Satisfying and Optimizing Metric
* In order to pick a good classifier we can use the average method to evaluate the performance of the classifier, but sometimes when the measurement criterias are not combinable then we use this kind of metic
* In this metric, we assign the measurement criterias as Optimizing and Satisficing
### Train/Dev/Test Distributions
Dev set is also called the hold out cross validation set
Choose a dev set and test set to reflect data you expect to get in the future and consider important to do well on
The dev set and the test set should come from the same distribution
Setting the dev set and the validation metric really defines what target you want to aim at
### Size of dev and test sets
Old way of splitting data(works with smaller sets)

Option | Training Set | Dev Set | Test Set
:-- | :--: | :--: | :--: 
Option 1 | 70% | 0% | 30%
Option 1 | 60% | 20% | 20%

* Set your test set to be big enough to give high confidence in the overall performance of your sys( 10000 - 100000 maybe)
* Sometimes train set and dev set are enough and test set is not even created(btu this is not recommended)
* If you have a set of 1M data then 98% of it should be split into dev set and % to test set
* And the rule of thumb is really to try to set the dev set to big enough for its purpose, which helps you evaluate different ideas and pick this up from AOP better. And the purpose of the test set is to help you evaluate your final cost. 
### When to change the dev/test set and metric?
Sometimes the Metric + Dev set preferes algo A but the user and the creator prefers algo B according to the working, environment, etc. in this case you should change metrics
Error =     
![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1603).png) 

A better error metric would be =     
![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1604).png) 

* So the guideline is, if doing well on your metric and your current dev sets or dev and test sets' distribution, if that does not correspond to doing well on the application you actually care about, then change your metric and your dev test set.In other words, if we discover that your dev test set has these very high quality images but evaluating on this dev test set is not predictive of how well your app actually performs, because your app needs to deal with lower quality images, then that's a good time to change your dev test set so that your data better reflects the type of data you actually need to do well on.
* Do not run for too long without any evaluation metric and dev set up because that can slow down the efficiency of what your team can iterate and improve your algorithm.
## Comparing to human-level performance
* System Progress tends to be relatively rapid as you approach human level performance. But then after a while, the algorithm surpasses human-level performance And the hope is it achieves some theoretical optimum level of performance.The performance approaches but never surpasses some theoretical limit, which is called the Bayes optimal error. So Bayes optimal error, think of this as the best possible error.
* When ML is worse than human, you can:
  * Get labeled data from humans
  * Gain insights from manual error analysis as of why the human got it right
  * Better analysis of bias/variance
### Avoidable Bias
Human level error is a proxy for Bayes error    
![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1605).png) 

### Understanding human level performance    
![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1606).png) 

So just to summarize what we've talked about. If you're trying to understand bias and variance where you have an estimate of human-level error for a task that humans can do quite well, you can use human-level error as a proxy or as a approximation for Bayes error.And so the difference between your estimate of Bayes error tells you how much avoidable bias is a problem, how much avoidable bias there is. And the difference between training error and dev error, that tells you how much variance is a problem, whether your algorithm's able to generalize from the training set to the dev set. And the big difference between our discussion here and what we saw in an earlier course was that instead of comparing training error to 0%,And just calling that the estimate of the bias. In contrast, in this video we have a more nuanced analysis in which there is no particular expectation that you should get 0% error. Because sometimes Bayes error is non zero and sometimes it's just not possible for anything to do better than a certain threshold of error.    
![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1607).png) 

### Surpassing human level performance
Eg:     
![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1608).png) 

What will the avoidable bias be in this?     
![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1609).png) 

Variance is the avoidable bias

Eg 2:     
![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1610).png) 

* once you've surpassed this 0.5% threshold, your options, your ways of making progress on the machine learning problem are just less clear. It doesn't mean you can't make progress, you might still be able to make significant progress

* Problems where ML surpasses human level performance
  * Online advertising
  * Product recommendations
  * logistics(predicting transit time)
  * Loan approvals
* All problems types that will surpass human level performance are:
  * ML algos that are learning from structured data
  * No natural perception problem
### Improving your model performance
* Two fundamental assumptions of supervise learning
  * You can fit training set pretty well
  * The training set performance generalizes pretty well to the de/test set
* How to avoid bias
  * Train bigger model
  * Train longer/better optimization algo
  * NN architecture/hyperparameters search
* Where variance is the problem
  * Use more data
  * Regularization
  * NN architecture/hyperparameters search
## Error Analysis
### Carrying out error analysis
* If your learning algorithm is not yet at the performance of a human, then manually examining mistakes that your algorithm is making, can give you insights into what to do next. This process is called error analysis
* Error Analysis method: 
  * Get ~100 mislabeled dev set examples
  * Solve the problem yourself and compare the results with the outcome the algo
### Cleaning up Incorrectly labelled data
* When you have the wrong input data. In this case you have to perform the following things
  * Look at the Training set: DL algorithms are quite robust to random errors in the training set. If the error is random it probably is low in number so this can be ignored. But if it is systematic(for eg: a white colored animal is always labelled as a dog even tho its a cat) then the labelled inputs should be dealt with
  * Look at the Dev Set: Create a table like this and mention in the column about the incorrectly labelled images
if the % of total error is less then do not look into this matter as it will be a waste of your time    

![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1624).png)    

* Correcting incorrect dev/text set examples
  * Apply same process to your dev and test sets to make sure they continue to come from the same distribution
  * Consider examining examp.es your algo got right as well as ones it got wrong
  * Train and dev/test saa may now come from slightly different distributions
### Build your first System quickly then iterate
Set up a dev/test set and metric and then build the initial system quickly. Then use Bias/Variance analysis and error analysis to prioritize next step
## Mismatched Training and Dev/Test Sets
### Training and Testing on different distributions
* Option 1: combine all the different distributions and then make one huge set to divide into the training and test set. 
Eg: if we have 200000 inputs from one distribution and 10000 from the other, then we can shuffle both to make a huge set of 210000 inputs and then divide this into 200000 for training set, 5000 for dev set and 5000 for test set
Advantage: all the sets come from same distribution
Disadvantage: The most of the dev and test set inputs will come from the first distribution that is the one with 200000 inputs
* Option 2(better option): keep the entire training set with inputs from 200000 set + a few from the 10000 input set and the dev and test set only contain inputs from the 10000 input set
Advantage:  you're now aiming the target where you want it to be
Disadvantage: your training distribution is different from your dev and test set distributions
### Bias and Variance training and dev/test data distribution
* It can be that the training error is 1% and the dev error is 10% which creates a huge gap. This gap can be due to two things. 
  * One is that the algorithm saw data in the training set but not in the dev set. 
  * Two, the distribution of data in the dev set is different
* In the above situation a Training-dev set can be created: same distribution as training set but not used for training. Now the training error remains the same as 1% and the new training-dev error is 9%. The dev error was still 10%. This shows us that there is a variance problem here because the training-dev error was measured on data that comes from the same distribution as your training set. So you know that even though your neural network does well in a training set, it's just not generalizing well to data in the training-dev set which comes from the same distribution.
* Let's look at a different example. Let's say the training error is 1%, and the training-dev error is 1.5%, but when you go to the dev set your error is 10%. So now, you actually have a pretty low variance problem, because when you went from training data that you've seen to the training-dev data that the neural network has not seen, the error increases only a little bit, but then it really jumps when you go to the dev set. So this is a data mismatch problem, where data mismatched. So this is a data mismatch problem,    

![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1627).png)    

* Bias/variance on mismatches training and de/test sets   

![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1628).png)   

* Sometimes instead of increasing the error from Human level to test error the error can decrease too
  * Example    

![image](https://github.com/DevNation-Community/structuring-machine-learning-projects/blob/main/Screenshot%20(1629).png)    

### Addressing data mismatch
* Carry out manual error analysis to try to understand difference between training and dev/test sets
* Make training data more similar; or collect more data similar to dev/test sets

## Learning from multiple tasks
### Transfer learning
* Sometimes you can take knowledge the neural network has learned from one task and apply that knowledge to a separate task. This is called transfer learning
* When does transfer learning make sense? 	
  * Transfer learning makes sense when you have a lot of data for the problem you're transferring from and usually relatively less data for the problem you're transferring to.
  * Task A and B have the same input x
  * Low level features from A could be helpful for learning B
### Multi Task Learning
* In multi-task learning, you start off simultaneously, trying to have one neural network do several things at the same time. And then each of these tasks helps hopefully all of the other tasks. 
* When multi-task learning makes sense
  * Training on a set of tasks that could benefit from having shared lower level features
  * Usually: amount of data you have for each task is quite similar
  * Can train a big enough neural network to do well on all the tasks
## End-to-end deep learning
* There have been some data processing systems, or learning systems that require multiple stages of processing. And what end-to-end deep learning does, is it can take all those multiple stages, and replace it usually with just a single neural network
* If you're training on 3,000 hours of data to build a speech recognition system, then the traditional pipeline, the full traditional pipeline works really well. It's only when you have a very large data set, you know one to say 10,000 hours of data, anything going up to maybe 100,000 hours of data that the end-to end-approach then suddenly starts to work really well
Pros of end-to-end learning
Lets the data speak
Less hand designing of components needed
Cons
May need large amount of data
Excludes potential useful hand designing components
