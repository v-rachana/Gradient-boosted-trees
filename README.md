# Gradient Boosted Trees

## How to Run the Boosting Trees Model

This guide provides step-by-step instructions on how to set up and run the Gradient Boosted Trees (GBT) model for classification and regression tasks using Python. This example uses the Iris dataset for classification and an example Concrete Data dataset for regression.

### Prerequisites

Ensure you have the following installed:

- Python 3
- Pandas
- NumPy
- Matplotlib
- pickle (for model serialization)

### Setup and Running

#### Step 1: Install Required Libraries

```bash
pip install numpy pandas matplotlib
```

#### Step 2: Running the Model

```bash
python test/gbt.py
```

## Model Training and Evaluation

The script provides multiple options for training and evaluating models:

### Train and Save the Iris Model

- **Description**: This option trains the GBT model on the Iris dataset and saves the trained model to disk.
- **How to Run**: To run this option, choose `1` when prompted.

### Load and Plot Iris Model

- **Description**: This option loads a previously saved Iris model and plots predictions against true values.
- **How to Run**: To run this option, choose `2` when prompted.

### Train Concrete Data Model

- **Description**: This option trains the GBT model on a concrete dataset (default `Concrete_Data.xls`) and saves the model.
- **How to Run**: To run this option, choose `3` when prompted.

### Train Custom Dataset

- **Description**: This option allows you to specify a custom dataset for model training.
- **How to Run**: To run this option, choose `4` when prompted and provide the filename (`test/Concrete_Data.xls`)

## Exiting the Program

- **How to Exit**: To exit the program, type `q` at the main menu prompt.


## 1. Boosting Trees

## Q1.What does the model you have implemented do and when should it be used?

This project implements Gradient Boosted Trees (GBT) in Python. 
GBT is a robust ensemble learning technique that sequentially builds decision trees, where each new tree corrects the errors of the previous ones. 

It's widely used for both:

a. Regression tasks (e.g., predicting concrete compressive strength)

b. Classification tasks (e.g., classifying flower species using the Iris dataset)

The core idea is:

“Fit a model, look at the errors, fit a new model to correct the errors, and repeat.”

GBT is adaptive, meaning each tree is dependent on the previous one. Unlike bagging methods (like Random Forests), boosting places more focus on difficult-to-predict instances.

## GBT is widely used in:

a. Finance: credit scoring, fraud detection.

b. Healthcare: disease prediction, risk modeling.

c. Marketing: churn prediction, customer segmentation.

d. Engineering: quality control, forecasting.

## When to Use This Model

Use this GBT implementation when:

a. You are dealing with non-linear relationships in your data.

b. You need better performance than a single decision tree.

c. You want to tune model behavior with hyperparameters like learning rate, number of estimators, or tree depth.

d. Your dataset is small to moderately sized and fits in memory.

## Q2.How did you test your model to determine if it is working reasonably correctly?

In machine learning, testing ensures the model generalizes well to unseen data. 
Overfitting occurs when the model learns noise instead of the signal. 
We used hold-out validation (train/test split) to evaluate our model's generalization performance.

##Testing on Iris Dataset (Classification)

a. Used Iris dataset with three classes.

b. Split into training/testing sets (80/20).

c. Trained GBT with default parameters.

d. Rounded regression predictions to the nearest class label.

e. Evaluated using accuracy.

f. Achieved classification accuracy > 90%.

g. Visualized predictions vs. true values.

## Testing on Concrete Dataset (Regression)

a. Used Concrete_Data.xls to predict compressive strength.

b. Evaluated using:

    i. RMSE (Root Mean Squared Error)

    ii.R² Score (explained variance)

c. Achieved RMSE < 10 and R² > 0.8.

d. Visual Validation
   
   i. Plotted predictions vs. true values.

  ii. Analyzed residual trends (error consistency).

 iii. Confirmed model’s behavior matches theoretical expectations.

 
## Q3.What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)

## Exposed Parameters for Performance Tuning

Our implementation provides several tunable parameters for users to adjust the model's behavior and performance.

## Why Parameter Tuning Matters

Hyperparameters are key levers in GBT that affect bias-variance tradeoff:

i. More trees -> Lower bias, but potential overfitting.

ii. Lower learning rate -> Higher accuracy, but slower convergence.

iii. Deeper trees -> More flexibility, but higher risk of overfitting.

### Parameters and Their Effects

#### The following parameters are customizable when initializing the GBT model:

| Parameter        | Description                                                                                | Default Value | Effect on Model                                                                                                        |
| ---------------- | ------------------------------------------------------------------------------------------ | ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `num_estimators` | Number of boosting rounds                                                                  | 20            | More trees can improve performance but increase computation time.                                                      |
| `max_depth`      | Maximum depth of each regression tree                                                      | 5             | Controls complexity; deeper trees may overfit.                                                                         |
| `min_split`      | Minimum samples required to split a node.                                                  | 2             | Higher values reduce overfitting.                                                                                      |
| `learning_rate`  | Learning rate (shrinkage).                                                                 | 0.01          | Lower values make training slower but often more accurate.                                                             |
| `criterion`      | Loss function: 'mse' (default) or 'mae'                                                    | ‘mse’         | mae is more robust to outliers.                                                                                        |

#### RegressionTree (Used Internally by GBT) Parameters

| Parameter           | Description                                                   | Default Value | Effect on Model                                                      |
| ------------------- | ------------------------------------------------------------- | ------------- | -------------------------------------------------------------------- |
| `max_depth`         | Maximum depth of the decision tree.                           | 5             | Controls the depth of individual trees. Similar to max_depth in GBT. |
| `min_samples_split` | Minimum number of samples required to split an internal node. | 2             | Avoids over-partitioning of data, ensuring meaningful splits.        |
| `criterion`         | Loss function for evaluating splits (mse or mae).             | ‘mse’         | Affects how split quality is determined.                             |

### Basic Usage Examples

#### Example 1: Classification on Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = GBT(num_estimators=30, max_depth=3, min_split=5, learning_rate=0.1, criterion='mse')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_pred_class = np.round(y_pred).astype(int)
y_pred_class = np.clip(y_pred_class, 0, 2)
accuracy = accuracy_score(y_test, y_pred_class)

print(f"Accuracy on Iris Dataset: {accuracy:.2f}")
```

## Dataset
The Iris dataset is a well-known multiclass classification dataset with three flower classes (Setosa, Versicolor, Virginica). It has continuous input features (petal and sepal dimensions) and categorical output.

## Why GBT Works Here
Even though the GBT implementation is originally built for regression, it can still be used for classification by:

Treating the classes as numerical values (0, 1, 2)

Using regression to predict a continuous value

Rounding the prediction to the nearest class

This is a clever workaround for classification using regression algorithms. However, it may not scale well to more complex classification tasks unless we implement softmax and cross-entropy loss.

## Interpretation of Evaluation
Accuracy measures how many of the rounded predictions match the actual class labels. If the model performs well, we expect accuracy > 90%, which was achieved in our tests.

### Example 2: Regression on Concrete Dataset

   ```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_excel('./Concrete_Data.xls')
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = GBT(num_estimators=50, max_depth=4, min_split=10, learning_rate=0.05, criterion='mse')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE on Concrete Data Dataset: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
   ```

## Dataset
The Concrete dataset is a regression dataset where the target is the compressive strength of concrete. Predictors include chemical ingredients like cement, water, slag, and age.

## Why GBT Works Well Here
GBT handles non-linearities and interactions between features better than linear models.

Boosting focuses on minimizing residuals, which is ideal when you want precise numerical predictions.

## Evaluation Metrics
RMSE (Root Mean Squared Error): Measures average prediction error. Lower is better.

R² Score (Coefficient of Determination): Measures how much variance in the target is explained by the model. R² closer to 1 indicates strong prediction.

## Model Settings
50 estimators: Gives the model enough learning iterations.

Learning rate 0.05: Conservative update step, better generalization.

Max depth 4: Balanced model complexity.

Criterion 'mse': Standard for squared error minimization.

### Example 3: Fine-Tuning the Model for Better Performance

```python
model = GBT(num_estimators=100, max_depth=6, min_split=15, learning_rate=0.02, criterion='mae')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Performance after fine-tuning: RMSE={rmse:.2f}")
```

## What Is Fine-Tuning?
Fine-tuning involves adjusting hyperparameters to strike the right balance between underfitting and overfitting. This process can significantly impact model performance.

## Why These Parameters?
i. num_estimators=100: More trees help the model learn finer patterns.

ii. learning_rate=0.02: Smaller steps prevent overshooting and improve generalization.

iii. max_depth=6: Allows for deeper decision rules without being too complex.

iv. min_split=15: Prevents splits on very small samples, reducing overfitting.

v. criterion='mae': MAE is robust to outliers because it treats all errors equally.

## What This Demonstrates
This example shows how you can customize the model to your dataset. Fine-tuning is essential in production-level machine learning to get the best out of your models.

## Q4.Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?

## Challenges and Workarounds

Our custom implementation of the Gradient Boosted Tree (GBT) model faces challenges with a few specific types of inputs and datasets. Below is a categorized analysis of these challenges, their workarounds, and whether they are fundamental to the implementation or solvable with additional development.

## Limits of Tree-Based Models

While tree models are flexible, they face issues with:

i. Scalability (large data)

ii. Missing values

iii. Categorical variables (non-numeric)

iv. Dynamic features or streaming data

### 1. Large Datasets

**Challenge:**

- Can be slow and memory-intensive due to full in-memory processing.

**Workaround:**

- Partial fix with dataset sampling.

**Fundamental:**

- Partially. Scaling requires significant optimizations, such as histogram-based boosting (used by libraries like XGBoost) and parallel processing for faster split evaluations.

### 2. High Dimensionality

**Challenge:**

- Slower split evaluations with many features.

**Workaround:**

- Apply dimensionality reduction (e.g., PCA).

**Fundamental:**

- No. Techniques like feature subsampling (as used in Random Forests) can improve efficiency for high-dimensional data.

### 3. Non-Numeric Targets

**Challenge:**

- Cannot handle strings or multi-label targets.

**Workaround:**

- Convert categories to integers.

**Fundamental:**

- No. Extending the implementation to handle multi-label or other target formats is feasible with additional development.

### Inputs That Are Fundamentally Challenging

Some inputs pose intrinsic challenges to the decision tree-based structure of the GBT model:

#### Dynamic Features

**Challenge:**

- Datasets where the number of features changes between training and inference.

**Workaround:**

- Preprocess data to ensure feature consistency across all stages.

**Fundamental:**

- Yes. This is a limitation of decision trees and ensemble methods in general.

#### Streaming Data

**Challenge:**

- Batch-learning only (no online/real-time updates).

**Workaround:**

- Requires architectural changes.

**Fundamental:**

- Yes. Adapting GBT for streaming scenarios would require architectural rework.

### Given More Time, What Could Be Addressed?

With additional time and effort, the following improvements could enhance the implementation:

- **Missing Values:** Implement surrogate splits or internal imputation during tree-building.
- **Large Datasets:** Optimize tree-building using histogram-based boosting and implement parallel processing for faster computation.
- **Robust Loss Functions:** Incorporate options like Huber loss to handle outliers.
- **High Dimensionality:** Introduce feature subsampling or regularization techniques.

### Conclusion

The Gradient Boosted Trees (GBT) implementation has demonstrated strong performance on both the Concrete Data and Iris datasets, showcasing its versatility in handling regression and classification tasks. 
However, while the current model is effective for many standard use cases, there are areas where it could be further improved to handle more complex datasets and overcome certain limitations.

### **1.GBT Predictions vs True Values on Concrete Data**

![GBT Predictions vs True Values](https://github.com/Spring2025CS584/Project2/blob/3c4e751c743c31357a6eeaddf263bd5abbd69182/images/GBT_Predictions_Concrete_Data.png?raw=true)

## Key Findings from the Plot:
a. The plot comparing true values (blue dots) and predicted values (red crosses) provides a clear visual assessment of the model's predictive accuracy.

b. In many instances, the predicted values closely align with the true compressive strength values, indicating that the GBT model captures significant trends within the data.

c. However, discrepancies between the true and predicted values in some samples highlight potential weaknesses:
   i. Certain predictions deviate significantly, suggesting that the model may struggle with specific patterns or outliers in the data.
  ii. These deviations emphasize the need for further optimization, such as hyperparameter tuning or incorporating more robust loss functions like Huber loss.

## Evaluation Metrics:

a. RMSE: The Root Mean Squared Error quantifies the average magnitude of prediction errors. A lower RMSE (< 10) confirms that the model performs well overall but leaves room for improvement.

b. R² Score: An R² score greater than 0.8 indicates that the model explains a substantial portion of the variance in the target variable. However, achieving an even higher R² score could enhance the model’s explanatory power.

##Conclusion for Concrete Data:

While the GBT model performs admirably, capturing most trends in the concrete compressive strength dataset, there are opportunities to refine its accuracy. 
Fine-tuning hyperparameters, exploring feature engineering techniques, or experimenting with alternative models could address remaining gaps and improve performance further.

---

### **2.GBT Predictions on Iris Dataset**

![GBT_Predictions_Iris](https://raw.githubusercontent.com/Spring2025CS584/Project2/3c4e751c743c31357a6eeaddf263bd5abbd69182/images/GBT_Predictions_Iris.png)

## Key Findings from the Plot:

a. The plot comparing true class labels (blue dots) and predicted class labels (red crosses) visually represents the model's classification accuracy.

b. In several areas, the red crosses perfectly overlap with the blue dots, indicating accurate predictions for those samples.

c. However, misclassifications occur in some instances, particularly when distinguishing between classes 1 (Versicolor) and 2 (Virginica), which are known to have overlapping features.

## Evaluation Metric:

Accuracy Score: The model achieves an accuracy greater than 90%, demonstrating its effectiveness in classifying the Iris dataset. 
However, the misclassifications suggest that further refinement could enhance performance, especially for borderline cases between similar classes.

## Conclusion for Iris Dataset:
The GBT model shows excellent performance in classifying the Iris dataset, particularly for the easily separable Setosa class. 
However, challenges remain in distinguishing between Versicolor and Virginica, which could be addressed through hyperparameter tuning or more advanced techniques like softmax and cross-entropy loss for multi-class classification.

### **Overall Observations and Recommendations**

## Strengths of the GBT Implementation:

a. Versatility: The model successfully handles both regression (Concrete Data) and classification (Iris) tasks, demonstrating its adaptability.
b. Performance: High accuracy (>90%) on the Iris dataset and low RMSE (<10) with a strong R² score (>0.8) on the Concrete Data confirm its robustness.
c. Tunable Parameters: Exposed hyperparameters like num_estimators, max_depth, learning_rate, and criterion allow users to fine-tune the model for optimal performance.

## Challenges and Areas for Improvement:

a. Scalability: The model struggles with large datasets due to memory constraints and slow processing times. Implementing histogram-based boosting and parallel processing could address this limitation.

b. High Dimensionality: While the current implementation handles moderate-dimensional data well, feature subsampling or dimensionality reduction techniques could improve efficiency for high-dimensional datasets.

c. Non-Numeric Targets: Extending the model to handle categorical or multi-label targets would broaden its applicability.

d. Streaming Data: The batch-learning nature of the current implementation limits its ability to handle dynamic or streaming data, requiring architectural changes for real-time updates.

## Future Work:

a. Optimization for Large Datasets: Incorporate histogram-based boosting and parallel processing to enhance scalability.

b. Handling Missing Values: Develop strategies for missing data, such as surrogate splits or internal imputation.

c. Robust Loss Functions: Introduce options like Huber loss to improve robustness against outliers.

d. Feature Engineering: Use techniques like PCA or feature subsampling to manage high-dimensional data more efficiently.

The GBT implementation is a powerful tool for both regression and classification tasks, excelling in capturing non-linear relationships and interactions between features. 
Its performance on the Concrete and Iris datasets highlights its strengths while also revealing areas for improvement. 

By addressing scalability, handling high-dimensional data, and extending its capabilities to more complex inputs, the model can become even more robust and versatile, capable of tackling a wider range of machine learning challenges.

