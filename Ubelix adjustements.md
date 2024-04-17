# Code modification and transfer to UBELIX
This notebook describes the process of modifying the code which was used in the "Documented_tutorial" notebook.

### 1. Creating the code
First all the code fragments of the notebook were combined. This combined code was then safed in a .py file. This can be done by copying the code from the notebook into a file created in the nano editor in the terminal or by using an IDE like Visual Studio Code. This .py file is then safed.

### 2. Adjusting the code
The combined code needs to be adjusted to be run in a non-notebook environment:
- Remove %matplotlib inline: This command is specific to Jupyter Notebook or IPython and is not required in a script running in a terminal.
- Save Plots to Files: Instead of displaying the plots interactively, save them to files using plt.savefig().
- Adjust the Plotting Code: Modify the plotting code to save the plots and remove the interactive display commands.

### 3. Run the code locally
To check if the code was adjusted correctly to be run in the terminal, first a new virtual environment was created and activated localy:
```bash
python -m venv testVenv
source testVenv/bin/activate
```

Then the following packages were installed in the vitrual environment:
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow
- rdkit
- seaborn
- setuptools

```bash
pip install pandas numpy scikit-learn matplotlib tensorflow rdkit seaborn setuptools
```

As this test was successfull it was time to transfer everything to UBELIX. 

### 4. Transfer to UBELIX
First a requirements file with the necessary python packages was created:
```bash
python -m pip freeze > requirements.txt
```

Then the a ssh-session was opened:
```bash
ssh username@submit03.unibe.ch
```

Then the python module was loaded:
```bash
module load Python
```

Then a virtual environment was created for the UBELIX user and activated:
```bash
python -m venv venvML
source venvML/bin/activate
```

Now the necessary files could be transferred to the created directory on UBELIX. The following 3 files were important:
- kinase.csv: data
- kinaseML.py: the python script
- requirements.txt: the necessary python packages

The file transfer was done with scp:
```bash
scp /Users/user1/testVenv/MidTermProject/* username@submit03.unibe.ch:~/venvML
```
![connection lost screenshot](https://github.com/orsli/MidTermProject/assets/160760991/1f193e49-40e1-48cc-8004-62c980994623)
As I got a "connection timed out, connection lost" error I transferred the files with FileZilla.
![FileZilla screenshot](https://github.com/orsli/MidTermProject/assets/160760991/ee98a8c6-eab6-4673-9726-f3f90343e3ec)


### 5. Prepare UBELIX
Now that all files were in the right place the necessary python packages could be installed in the virtual environment on UBELIX:
```bash
pip install -r requirements.txt
```

This step took very long and was aborted two times. There seems to have been an issue with the tensorflow package. Therefore the tensorflow package was then installed manually on its own:
```bash
pip install tensorflow
```

Trying to run the python code in this virtual environment via the slurm job failed multiple times. So the installation of the packages was repeated without the virtual environment, just as the standard UBELIX user.

### 6. Test of a SLURM Job
To test the job-submission system a short script was written:
```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1GB

# Put your code below this line
python test.py > test.txt
```

This test was successful. Therefore the final step was to create a submission script for the correct ML python code:
```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=4GB

# Run your Python script
python kinaseML.py > outputML.txt
```

As specified in the code, the output was saved to the same venvML folder as the python code file.
As a last step the output was transferred back to the local computer:
```bash
scp username@submit03.unibe.ch:~/venvML/scatterplot.png ~/Desktop
scp username@submit03.unibe.ch:~/venvML/training.png ~/Desktop
scp username@submit03.unibe.ch:~/venvML/outputML.txt ~/Desktop
```

The following two pictures show the training and the test-prediction of the ML model:
<img src="https://private-user-images.githubusercontent.com/160769363/322268584-460f48a7-baf4-4177-8170-8736570c2863.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTMwOTM2OTcsIm5iZiI6MTcxMzA5MzM5NywicGF0aCI6Ii8xNjA3NjkzNjMvMzIyMjY4NTg0LTQ2MGY0OGE3LWJhZjQtNDE3Ny04MTcwLTg3MzY1NzBjMjg2My5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDE0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQxNFQxMTE2MzdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yNjMyOTU1YzM3ODY0MjgzY2Y3MjZhNTQ1NzBmZjA3OWRiYzhjNjZmNWI0OWMwYTE0M2Q3Y2MwNjkzZTdjMjA5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.Hr3c04uevKxzpZscVedQ7wwJGyCBP0uLqFrGKi0-q2w" width=80% height=80%>
<img src="https://private-user-images.githubusercontent.com/160769363/322268581-892c5160-87fd-468a-bb0f-f92f1547f020.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTMwOTM2OTcsIm5iZiI6MTcxMzA5MzM5NywicGF0aCI6Ii8xNjA3NjkzNjMvMzIyMjY4NTgxLTg5MmM1MTYwLTg3ZmQtNDY4YS1iYjBmLWY5MmYxNTQ3ZjAyMC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNDE0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDQxNFQxMTE2MzdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lN2Y2NGVmNDEzZWZhMjRlYzhiY2Y2ZjdhOWU4NGZiODYwNTRmNjM5N2M3OWEzNTEyNDQwNTUwNWUxNzRhYmVhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.84M8XmTVglPt6lf25IrWLNDLEnyq1OAC1Zj-QCiDPPI" width=40% height=40%>
