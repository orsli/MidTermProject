## 1. What is Ubelix?
Ubelix, short for "University of Bern Linux Cluster," represents the University of Bern's high-performance computing (HPC) system. HPC involves using powerful computational resources to solve complex problems or process large amounts of data quickly. It typically consists of interconnected clusters of computers with multiple processors, enabling parallel processing for tasks like simulations, modeling, and data analysis across various fields such as science, engineering, and industry. HPC accelerates scientific discoveries and facilitates solving complex challenges that traditional computing cannot handle efficiently.

## 2. How do you gain access?
UBELIX is accessible to University of Bern members via a valid Campus Account (CA). Here's how to get access:
- Activation Request: Submit a request at https://serviceportal.unibe.ch/hpc, providing your CA username and a brief description of your intended use.
- Username Details: Your UBELIX username and password will match your Campus Account credentials.

For external collaborators:
- The institute's account manager should request a Campus Account from the University of Bern's IT department.

## 3. How do you submit a job?

To submit a job on UBELIX, follow these steps:

#### 1. Login to the Cluster

- Connect to the cluster by logging in to a login node from within the university network. Use a secure shell (SSH) to log in to one of the available login nodes (e.g., submit03.unibe.ch):
  ```
  ssh <user>@submit03.unibe.ch
  ```
  After successful login, you'll find yourself in the directory /storage/homefs/$USER, where $USER is your Campus Account username.

#### 2. Prepare Your Job

- Before submitting a job, ensure your files are in the correct directory. If you need to copy files between your local computer and the cluster, you can use the secure copy command scp. To copy a file from your local computer running a UNIX-like OS use the secure copy command scp on your local workstation:
  ```bash
  scp /path/to/file <user>@submit03.unibe.ch:/path/to/target_dir/
  ```
  To copy a file from the cluster to your local computer running a UNIX-like OS also use the secure copy command scp on your local workstation:
  ```bash
  scp <user>@submit03.unibe.ch:/path/to/file /path/to/target_dir/
  ```

#### 3. Write Your Job Script

- Create a job script containing instructions for the scheduler. Define resource requirements and specify tasks to be executed. For example:
  ```bash
  #!/bin/bash
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=1
  #SBATCH --mem-per-cpu=1GB

  # Put your code below this line
  ```
  This script allocates one task, one CPU per task, and 1GB of memory per CPU.

#### 4. Execute Your Job

- Submit your job to the scheduler using the sbatch command:
  ```bash
  sbatch your_job_script.sh
  ```
  If the submission is successful, you'll receive a job ID to reference your job later.

#### Additional Information

- Explore various options for different job types provided by the scheduler, such as Array Jobs, GPUs, and Interactive Jobs. Refer to the documentation for more details.

## 4. Who can have access?

1. University of Bern Members:
- All individuals with a valid Campus Account (CA) affiliated with the University of Bern.
- Access is primarily granted for research aligned with the university's interests.

2. External Collaborators:
- External researchers collaborating with a University of Bern institute.
- Access is facilitated through the account manager of the respective institute, who must request a Campus Account from the University of Bern's IT department on behalf of the external collaborator.

## 5. What resources are available there?

Ubelix provides an impressive array of computational resources. With around 320 compute nodes housing approximately 12,000 CPU cores and 160 GPUs, it offers substantial computing power. Users have access to a variety of pre-installed software and applications, with the flexibility to install custom software. This rich resource pool empowers users to tailor their computing environment to their needs, driving innovation and research across disciplines.
