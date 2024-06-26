## 1. What is in the training set, how big is it?
The training set consits of 179'827 entries of data related to molecules and their properties. To perform the tutorial only the two columns "standard_value" and "smiles" are needed.
* molecule_chembl_id: The ChEMBL database identifier for the specific molecule.
* standard_value: The standardized potency or concentration ot the molecule in terms of IC50 (half maximal inhibitory concentratin).
* standard_units: The units in which the standard value is expressed.
* target_chembl_id: The ChEMBL database identifier for the target associated with the molecule.
* smiles: The simplified molecular-input line-entry system (SMILES) notation which expresses the inline textual representation of the molecule's structure.

## 2. What modifications do you need to do to the data set to perform the tutorial.
To be able to use the data set for performing the tutorial, the following modifications had to be done:
1. **Calculating the pIC:** For each entry in the database we calculated the negative logarithm of the IC values and stored them in a new column. The pIC was calculated by the formula pIC = -log(IC).
2. **Generating the MACCS fingerprints:** The fingerprints are representations of the molecular structure which can then be used for machine learning tasks in cheminformatics. For that, we followed the instructions in the [tutorial](https://projects.volkamerlab.org/teachopencadd/talktorials/T022_ligand_based_screening_neural_network.html).
3. **Extracting the important columns:** We extracted the important columns from the dataset into a new file, such as the pIC values and the MACCS fingerprints.

## 3. What is a test set? Any other types of set?
A test set is a separate dataset used to evaluate the performance of a machine learning model trained on the training set. In the context of molecule properties prediction or drug discovery, a test set would likely contain similar information to the training set, such as molecule identifiers, properties, and structural information. However, the molecules in the test set would not have been used to train the model. Instead, they are reserved specifically for assessing how well the trained model generalizes to new, unseen data. In case of the given tutorial, the test set is derived from the same dataset as the training set. The original dataset is split into training and test set before the training.

Besides the training and test sets, there are also validation and prediction sets:
* Validation sets are additional datasets used during the model training process to tune hyperparameters and assess model performance during training. They are typically used alongside the training set to optimize the model's performance and prevent overfitting. Additionally, in some cases, there might be datasets specifically curated for tasks like cross-validation, where the training set is divided into multiple subsets for training and validation iteratively.
* The prediction set, also known as the inference set or the deployment set, is a dataset used for making predictions using a trained machine learning model. This set typically contains data similar in structure to the training and test sets, but it does not include the ground truth labels or outcomes. Instead, it consists of input features or variables for which predictions are to be generated by the trained model. Once a machine learning model has been trained and evaluated on a training and test set, it is deployed or put into production to make predictions on new, unseen data, which is the prediction set. The performance of the model on this set helps assess its real-world applicability and effectiveness. The prediction set is crucial for applying the model to practical tasks, such as predicting properties of new molecules in drug discovery or making recommendations in recommender systems. It allows organizations to leverage the insights gained from the trained model to make informed decisions in various domains.

## 4. Before starting describe with 1-2 sentences, in your own words, what is done in each of the cells.
1. Import Libraries: The necessary libraries are imported, including pandas, numpy, RDKit, scikit-learn, matplotlib, seaborn, and TensorFlow.
2. Data Preparation: The cell sets the path to the notebook and loads the dataset containing information about compounds targeting kinases from a CSV file into a pandas DataFrame.
3. Molecular Encoding: A function is defined to convert SMILES strings to MACCS fingerprints, which are numerical representations of molecular structures used for machine learning.
4. Data Labeling: The SMILES strings in the dataset are converted to MACCS fingerprints using the defined function, creating a new column in the DataFrame. This step prepares the data for machine learning by encoding molecular structures as numerical features.
5. Define Neural Network Model: A function is defined to create a neural network model using Keras. The model architecture consists of two hidden layers with ReLU activation functions and a linear activation function on the output layer. The model is compiled with mean squared error loss and the Adam optimizer.
6. Train the Model: Different mini-batch sizes are tried, and the corresponding losses are plotted to select the optimal batch size. The model is trained using the selected batch size, and the weights giving the best performance are saved.
7. Evaluation & Prediction on Test Set: The trained model is evaluated on the test set to assess its performance using mean squared error and mean absolute error metrics. The model is then used to predict pIC50 values for compounds in the test set.
8. Scatter Plot: A scatter plot is created to visualize the predicted vs. true pIC50 values on the test set, evaluating the model's predictive capability.
9. *Prediction on External/Unlabeled Data: The trained model is used to predict pIC50 values for compounds in an external/unlabeled dataset. The predictions are saved, and the top three compounds with the highest predicted pIC50 values are selected for further investigation.*
10. *Select the Top 3 Compounds: The top three compounds with the highest predicted pIC50 values are selected, and their molecular structures are visualized using RDKit.*
*Steps 9 and 10 are not performed.*
