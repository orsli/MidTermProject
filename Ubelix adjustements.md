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

![training](https://github.com/orsli/MidTermProject/assets/160760991/b11b645c-b5f0-4ba9-9f90-75701562fab3)
![scatterplot](https://github.com/orsli/MidTermProject/assets/160760991/0931f5b0-8e8f-42ce-8521-5f2f8b1d0d3c)


