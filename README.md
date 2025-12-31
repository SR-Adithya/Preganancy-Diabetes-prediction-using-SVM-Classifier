# Preganancy-Diabetes-prediction-using-SVM-Classifier
Prediction of diabetes for women undergoing pregnancy using Support Vector Machine Classifier model

## About Dataset
Name: Pima Indians Diabetes. Columns Used: Age, Pregnancies, Glucose, BloodPressure, BMI, Diabetes Pedigree Function, Outcome. No of Rows: 768 records

## Model Selection and it's reason:
Model: Support Vector Machine Classification (Supervised Model)
Description:
•	The outcome column is a class variable affected by multiple independent factors.
•	The model learns the pattern for each factor trained to provide precise and anticipated results. Consequently, the system is quick and dependable.
•	The result i.e the outcome column is intended to be a single variable correlated with several factors, hence support vector machine classifier model was implemented.

## Training method of data:
•	Total Dataset used = 768 records
•	Test size were listed and used test_size = 0.20. Hence, 614 records were adopted as test case and 154 records were assigned in training the model.
•	Scaler used: StandardScaler. Standardizes multiple factors into a single value.
•	To assist the model with learning multiple possibilities, random_state 42 was implemented.
•	Kernel “poly” utilized as medical parameters interact non-linearly. This would plot the segments with curved boundaries comprising all the related features.
•	Metrics such as accuracy score, classification report, confusion matrix were imported to evaluate the performance and accuracy of the model.

## Metrics used for Evaluation:
Metrics used:
		Confusion matrix: 
	Confusion matrix represents a table visual that highlights number of predictions made correct and incorrect for each class by the model
	The values are count based representation and used for other evaluation metrics.
	| **Predicted \\ Actual** | **Positive (1)** | **Negative (0)** |
|-------------------------|------------------|------------------|
| **Positive (1)**        | TP               | FP               |
| **Negative (0)**        | FN               | TN               |


**Where:** 
-**\(TP (True positive )\)**=correct prediction of positive classes
-**\(TN (True negative )\)**=correct prediction of negative classes
-**\(FP (False positive )\)**=incorrect prediction of positive classes
-**\(FN (False negative )\)**= incorrect prediction of negative classes
	
True positive (TP) and true negatives (TN) provide the perfect right as right and perfect wrong as wrong prediction classes whereas false positive (FP) and false negative (FN) values show where the model predicts the right ones as wrong and vice versa.
Accuracy score: 
Used to measure the closeness of the measured value to the standard value. A single value that summarizes the whole model’s performance
The value is calculated by using the below formula:

$$
\text{Accuracy score} = \frac{TP+TN}{TP+TN+FP+FN}
$$

**Where:** 
-**\(TP (True positive )\)**=correct prediction of positive classes
-**\(TN (True negative )\)**=correct prediction of negative classes
-**\(FP (False positive )\)**=incorrect prediction of positive classes
-**\(FN (False negative )\)**= incorrect prediction of negative classes
	
The higher value of accuracy score marks the model’s performance at its best. Low accuracy score shows the model struggles to differentiate between classes. 
Classification report: 
Provides comprehensive performance analysis such as recall, precision, F1 score and shows the behavior of each classes.
Recall: Ratio of correct predicted positive among all actual positives

- Formula used: 

$$
\text{Recall} = \frac{TP}{TP+FN}
$$

Helps to capture exact positive cases

Precision: Ratio of correct predicted positives among all predicted positives
- Formula used:

$$
\text{Precision} = \frac{TP}{TP+FP}
$$

Useful to mitigate false positives

F1-Score: Utilizes precision and recall and calculates their mean harmonically to balance the false positives and negatives into a single value.
- Formula used:

$$
\text{F1-score} = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

Reduces false positives and false negatives

Evaluated Metric Values from the model:

Confusion matrix: 
	| **Predicted \\ Actual** | **Positive (1)** | **Negative (0)** |
|-------------------------|------------------|------------------|
| **Positive (1)**        | 25               | 9               |
| **Negative (0)**        | 30               | 90               |

  The contingency table utilized 154 test case record
  90 records were correctly predicted as negative (no diabetes) case, 9 false positive (have diabetes but no diabetes), 30 false negative (no diabetes but have diabetes), and 25 as correctly predicted as positive case (have diabetes)
  Hence, the false positive and false negative has to be reduced further.

Accuracy score = 0.75

Classification report:
Recall: 0.91 (for no diabetes), 0.45 (for having chance of diabetes)
Precision: 0.75 (for no diabetes), 0.74 (for having chance of diabetes)
F1 score: 0.82 (for no diabetes), 0.56 (for having chance of diabetes)

## Strengths and Weakness of the model:
Strength: The model is well suited for medical datasetsand works effectively in high-dimensional feature spaces. It can handle several clincal parameters without experiencing appreciable performance detrioration. Due to the presence of kernel, it is able to identify physiological patterns that might not be distinguishable with linear models. 
Weakness: SVM is computationally costly and not directly interpretable, particularly when trained on big datasets or with intricate kernel functions.

## Possible improvements / real-world applications:
Improvement: Apply feature scaling and normalisation, adjust hyperparameters methodically. To evaluate the relative efficacy of SVM, compare its performance with that of other classifiers.

real-world application: Standard clinical criteria for the early detection of gestational diabetes mellitus (GDM) in prenatal care. Integration with tools for tracking maternal health to provide ongoing risk assessment during pregnancy. Research instruments for examining trend in diabetes risk variables at the population level.

## Conclusion of the project:
In view of the above study, the support vector machine has successfully classified the existing data into the respective class variables and demonstrated the same for new input data. The skewed distributions of glucose and age levels showcased as one of the compulsory features while diagnosing diabetes for pregnant women. The plots represent significant patterns that emerged via exploratory data analysis, supporting its function as the main risk indicator. With an overall accuracy of 75%, the support vector machine classifier model may be used to predict diabetes risk in pregnant women, yet the model has to be monitored, verified by the specialist, and improvises whenever encountering an outlier data each time. 
