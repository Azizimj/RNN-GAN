# CSCI-599 Assignment 2

## The objectives of this assignment
* Implement the forward and backward passes as well as the neural network training procedure for Recurrent Neural Networks (RNNs)
* Learn the basic concepts of language modeling and how to apply RNNs
* Implement popular generative model, Generative Adversarial Networks (GANs)

## Work on the assignment
Please first clone or download as .zip file of this repository.

Working on the assignment in a virtual environment is highly encouraged.
In this assignment, please use Python `3.5` (or `3.6`).
You will need to make sure that your virtualenv setup is of the correct version of python.

Please see below for executing a virtual environment.
```shell
cd CSCI599-Assignment2
pip3 install virtualenv # If you didn't install it
virtualenv -p $(which python3) /your/path/to/the/virtual/env
source  /your/path/to/the/virtual/env/bin/activate

# Install dependencies
pip3 install -r requirements.txt

# install tensorflow (cpu version, recommended)
pip3 install tensorflow

# install tensorflow (gpu version)
# run this command only if your device supports gpu running
pip3 install tensorflow-gpu

# Work on the assignment
deactivate # Exit the virtual environment
```

## Work with IPython Notebook
To start working on the assignment, simply run the following command to start an ipython kernel.
```shell
# add your virtual environment to jupyter notebook
python -m ipykernel install --user --name=/your/path/to/the/virtual/env

# port is only needed if you want to work on more than one notebooks
jupyter notebook --port=/your/port/

```
and then work on each problem with their corresponding `.ipynb` notebooks.
Check the python environment you are using on the top right corner.
If the name of environment doesn't match, change it to your virtual environment in "Kernel>Change kernel".

## Problems
In each of the notebook file, we indicate `TODO` or `Your Code` for you to fill in with your implementation.
Majority of implementations will also be required under `lib` with specified tags.

### Problem 1:  (60 points)
The IPython Notebook `Problem_1.ipynb` will walk you through implementing a recurrent neural network (RNN) from scratch.

### Problem 2: Generative Adversarial Networks  (40 points)
The IPython Notebook `Problem_2.ipynb` will help you through implementing a generative adversarial network (GAN) using TensorFlow.

## PLEASE DO NOT CLEAR THE OUTPUT OF EACH CELL IN THE .ipynb FILES
Your outputs on the .ipynb files will be graded. We will not rerun the code. If the outputs are missing, that will be considered as if it is not attempted.

## How to submit

Run the following command to zip all the necessary files for submitting your assignment.

```shell
sh collectSubmission.sh
```

This will create a file named `assignment2.zip`, please rename it with your usc student id (eg. 4916525888.zip), and submit this file through the [Google form](https://goo.gl/forms/ZnGEMcsW9yULARju2).
Do NOT create your own .zip file, you might accidentally include non-necessary
materials for grading. We will deduct points if you don't follow the above
submission guideline.

## Questions?
If you have any question or find a bug in this assignment (or even any suggestions), we are
more than welcome to assist.

Again, NO INDIVIDUAL EMAILS WILL BE RESPONDED.

PLEASE USE **PIAZZA** TO POST QUESTIONS (under folder assignment2).
