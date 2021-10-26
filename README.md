# Manually-Trained-Logistic-Regression-Model
This project deals with implementing the Linear Regression model on **Titanic dataset** which is uploaded in this repository in two parts as Train set & Test set.

Python is highly advantageous in using the readily available functions for machine learning algorithms BUT this is in a way leading to a disadvantage for Artificial Intelligence enthusiasts who want to dig deep into details & learn the algorithms with critical optimization techniques.

In the above, I have implemented the Linear Regression model using the following **Numerical Optimization Techniques** & performed an analysis on them:
1. Gradient Descent
2. Stochastic Gradient Descent (Mini-batch)
3. Stochastic Gradient Descent with Momentum
4. Stochastic Gradient Descent with Nesterov Momentum
5. AdaGrad
6. Adam

In the above, I used Precision, Recall & F-Score for analysis.
- **Precision** means the percentage of the results which are relevant.
- **Recall** refers to the percentage of total relevant results correctly classified by the model.
- **F-Measure** provides a way to combine both precision and recall into a single measure that captures both properties. It is calculated as: **F-Measure = (2 * Precision * Recall) / (Precision + Recall)**

In the code, modifying the values of learning rate (alpha), number of iterations (before convergence), rho values etc. will change the accuracy values. You can play around with those values to make analysis. View the attached sample snapshots (not completely smoothed so that you can try out changing the parameters for better accuracies).

For conceptual understanding, you can watch: https://www.youtube.com/watch?v=nhqo0u1a6fw

Execution: **python Logistic-Regression.py** (Make sure to have dataset files in the same location as Logistic-Regression.py file)
