# Credit-Card-Fraud-Detection
This repository implements a machine learning solution for detecting credit card fraud. The project utilizes advanced data preprocessing, model building, and adversarial training techniques to create a robust model capable of identifying fraudulent transactions.

## Key Features

### Data Preprocessing

- **Data Loading & Inspection:** Load and explore the dataset structure.
- **Missing Values:** Handle missing data by removing rows or imputing values.
- **Feature Scaling:** Standardize numeric features like Amount and Time.
- **Feature and Target Separation:** Separate features (X) and target label (y).
- **Train-Test Split:** Split data with stratification to preserve class balance.
- **Class Imbalance Handling:** Apply SMOTE to generate synthetic samples for minority classes.
- **Optional Enhancements:** Outlier detection, feature engineering, dimensionality reduction (PCA, t-SNE).

### Model Building

- **Neural Network:** Custom architecture with multiple hidden layers and ReLU activation.
- **Binary Classification:** Sigmoid function used in the output layer.
- **Model Compilation:** Optimizer (Adam), loss function (binary crossentropy), and evaluation metrics (accuracy, AUC).
- **Class Weights:** Optionally adjust class weights to handle imbalance during training.
- **Hyperparameter Tuning:** Adjust learning rate, neurons, batch size for improved performance.

### Model Evaluation

- **Evaluation Metrics:** Accuracy, precision, recall, F1-score, AUC-ROC.
- **Confusion Matrix:** Analyze true/false positives and negatives.
- **AUC-ROC & Precision-Recall Curves:** Visualize model performance.
- **Explainability:** Understand feature importance and individual predictions.

### Fine-Tuning

- **Hyperparameter Optimization:** Use GridSearchCV or RandomSearch for optimal parameters.
- **Regularization:** L2 regularization and dropout to prevent overfitting.
- **Early Stopping:** Stop training when validation performance ceases to improve.

### Model Deployment

- **Real-Time Predictions:** User inputs transaction data for immediate fraud detection.
- **UI Features:** Confusion matrix, AUC-ROC, visualizations.
- **User Authentication:** Optional security feature for access.
- **Deployment:** Host the app on Streamlit Cloud or any other cloud provider.

### Advanced Enhancements

- **Adversarial Defense:** Techniques like adversarial training.

# Dataset

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# References

- Libraries used can be found in requirements.txt


# Authors
1. Prashanth S (BMSCE)
2. Yashas Nandan S (BMSCE)
3. Pavan Kumar S (BMSCE)
4. Suhaas R (BMSCE)
