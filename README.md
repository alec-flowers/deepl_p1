# deepl_p1
Project 1 - Classification, Weight Sharing, Auxiliary Losses

The project's main objective is to compare the digits in two handwritten digits from MNIST dataset.

The files here are:
1. `runner.py`: contains functions managing the training and testing of all the models. The specific classes for each network in this file handles any necessary data manipulating that the network needs.
2. `utils.py`: contains useful functions to evaluate the different models, plotting and reporting the outputs.
3. `Data.py`: functions used to load and generate the data.
4. `test.py`: run file built to the project specifications. The hyper-parameters are set in this file, e.g. learning rate is set to $1e^{-4}$ and the networks' size. some verbosity enumerator for dumping train summary and `Tensorboard` output are also set here. Running this file, as is, will initiaite all of the designed architecture twice and report the `mean` and `STD` of the train and test accuracy. More can be obtained by changing the input and hyper-parameters.
5. plots.py: On top of what `test.py` does, this file regenerates the plots presented in the report.
