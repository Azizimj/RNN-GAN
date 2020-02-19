

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
cd <wd>
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

### Problem 1:
The IPython Notebook `Problem_1.ipynb` will walk you through implementing a recurrent neural network (RNN) from scratch.

### Problem 2: Generative Adversarial Networks
The IPython Notebook `Problem_2.ipynb` will help you through implementing a generative adversarial network (GAN) using TensorFlow.
