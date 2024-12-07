# deep-learning-challenge
Overview of the Analysis
The purpose of this analysis is to build and optimize a deep learning model using TensorFlow to predict whether charitable organizations will be successful in securing funding. The model is trained on historical data, and its performance is evaluated to ensure that it meets or exceeds a target predictive accuracy of 75%.

Results
Data Preprocessing

What variable(s) are the target(s) for your model?
The target variable is IS_SUCCESSFUL, which indicates whether a funding application was successful (1) or not (0).
What variable(s) are the features for your model?
The features include all other variables in the dataset after removing the EIN and NAME columns, which do not contribute to the predictive analysis. Preprocessed categorical variables such as APPLICATION_TYPE and CLASSIFICATION are one-hot encoded to be included as features.
What variable(s) should be removed from the input data because they are neither targets nor features?
The EIN and NAME columns are removed because they are identifiers and do not contain information useful for the prediction.
Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Neurons:
Hidden Layer 1: 8 neurons
Hidden Layer 2: 5 neurons
Output Layer: 1 neuron
Layers:
Two hidden layers and one output layer were selected to balance model complexity and training efficiency.
Activation Functions:
ReLU (Rectified Linear Unit) was used for the hidden layers because it is effective for non-linear problems and reduces the likelihood of vanishing gradients.
Sigmoid was used for the output layer to predict probabilities for binary classification.
These parameters were selected as a starting point, and adjustments were made during optimization attempts.
Were you able to achieve the target model performance?
The initial model did not achieve the target accuracy of 75%.
Optimization attempts, including modifying the number of neurons and layers, increasing the epochs, and refining the input data, brought the accuracy closer to the target. However, achieving over 75% accuracy required significant model fine-tuning.
What steps did you take in your attempts to increase model performance?
Adjusted the number of neurons in the hidden layers to increase the model's capacity to learn patterns.
Added an additional hidden layer to allow the model to learn more complex relationships in the data.
Tried different activation functions such as tanh for hidden layers to evaluate their impact on performance.
Increased the number of epochs during training to give the model more opportunities to learn.
Refined the input data by adjusting the thresholds for categorical binning to reduce noise and improve feature representation.
Summary

Overall Results:
The deep learning model achieved near-target accuracy after optimization but struggled to consistently surpass 75%.
Optimizing the model required extensive experimentation with layers, neurons, activation functions, and preprocessing strategies.
Recommendations:
Alternative Model Approaches:
A Random Forest or Gradient Boosting model could be used to address this classification problem. These ensemble methods can handle categorical data and variable importance effectively and may achieve higher accuracy with less parameter tuning compared to deep learning.
Reason for Recommendation:
Ensemble models are well-suited for structured tabular data like this dataset. They can handle feature interactions effectively and are less prone to overfitting due to inherent regularization.
