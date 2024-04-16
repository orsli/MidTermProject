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

## 3. What is a test set? Any other types of set?


## 4. Before starting describe with 1-2 sentences, in your own words, what is done in each of the cells.
