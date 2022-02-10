# IA3 README: 
- Author: Oluwasegun Somefun, somefuno@oregonstate.edu
- Course: AI534, Fall2021

## Overview
This README contains instruction that aid in the attempt to reproduce the python environment that this assignment was run on exactly, most specifically to run on the Babylon servers
- The ##* prefix used here indicate a section heading.
- The $ prefix indicate a command to be entered in a terminal (e.g: bash).

## 1. Required Python Version: 3.6+
- Check you have Python 3.6+ Installed
$ ls -ls /usr/bin/python*

## 2. Directory Structure: 
- Name of Zipped Code Directory: 'iax'
- Assume the unzipped code directory name is 'iax'.
$ cd iax 
- Confirm the presence of a 'readme.txt' and 'requirements.txt' file
$ ls

## 3. Install, Activate a Python Virtual Environment (venv)
##      and Set python3.6+ as default python.
- Create a venv in the 'iax' directory. 
- For example: desired venv name could be 'ai534oas'
$ python3.6 -m venv ai534oas 
$ source ai534oas/bin/activate && alias python='ai534oas/bin/python3.6'

-- Optional: On Windows PowerShell, the equivalent is 
-- $ .\ai534oas\Scripts\activate   
-- Also, see https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/


## 4. Install Dependencies
$ pip install --upgrade pip
$ pip install -r requirements.txt

### 4.1. Confirm Installed Packages
$ pip list


## 5. Running this Assignment
- Recommended Approach: 
- Run the ia3_part*.ipynb Jupyter notebooks
- for interactive logging and visualization.

### 5.1. Core source-code functions can be found in the .src/ folder
 - preprocessing: 'preprocess.py'
 - ML models: 'archs_models.py'
 - inferencing: 'infer_models.py'
 - gradient-based optimization: 'opts_models.py'

### 5.2. According to the question prompts, this assignment 
	is implemented using 5 notebooks for the parts:
- ia3_part1.ipynb :  Averaged and Online Perceptron
- ia3_part2a.ipynb : Online Kernelized Perceptron
- ia3_part2a_c.ipynb : Online Kernelized Perceptron (Run-Time Analysis)
- ia3_part2b.ipynb : Batch Kernelized Perceptron
- ia3_part2b_c.ipynb : Batch Kernelized Perceptron (Run-Time Analysis)

### 5.3 Report
- See 'report_ia3.pdf' for the submitted report for this assignment

## 6. Optional
### 6.1. Leave the venv
$ deactivate 

### 6.2. Delete the venv
$ rm -rf ai534oas


### 6.3. Questions/Issues
- I typically reply within a day, E: somefuno@oregonstate.edu