# FYS-STK4155 Project 2
This is our source code for Project 2 in the course [FYS-STK4155 Applied Data Analysis and Machine Learning](https://www.uio.no/studier/emner/matnat/fys/FYS-STK4155/index-eng.html) at the University of Oslo.

The project is based on classification using logistic regression (LR) and a multilayer perceptron (MLP) on credit card data from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). Additionally, we will perform regression analysis on the two-dimensional Franke function and compare the results with a [prior analysis](https://github.com/bernharl/FYS-STK4155-project1) where we used ordinary least squares. 

The aim of this project is to get a deeper understanding of the two different methods and to apply them to a real data set. To acheieve this, we have made our own implementation of LR and MLP in Python. 

To generate/read data, train the models, print/plot the results and build the report, please run <tt> main_script.sh </tt>. 

## Source structure 

* <tt> src/main.py</tt>: Main script containing all regression met* hods 
* <tt> src/test_main.py</tt>: Unit tests for <tt> src/main.py </tt>.
* <tt> src/read_credit_data</tt>: Reads, preprocesses and exports the credit card data as  <tt>.npz</tt> files.
* <tt> src/generate_franke.py</tt>: Generates, preprocesses and exports the Franke function data as <tt>.npz</tt> files.
* <tt> src/train_*.py</tt>: Trains and exports the models
* <tt> src/plot_*.py</tt>: Plotting script for the logistic regression model, MLP classification model and MLP regression model. 
* <tt> src/data</tt>: Folder where all the data from <tt> src/generate_franke.py</tt> and <tt> src/read_credit_data.py</tt> is saved.
* <tt> src/models</tt>: Folder where the models for each data set is saved. 
* <tt> src/cv_results</tt>: Folder where all the optimalized hyperparameters found by cross-validation in <tt> src/train_*.py</tt> are saved.

* <tt> doc/report_2.tex</tt>, <tt> doc/references.bib</tt>: <tt> .tex</tt> file for the project report and references
* <tt> doc/report_2.pdf</tt>: The built report as a  <tt>.pdf</tt>.
* <tt> doc/figures</tt>: Folder where all the figures from <tt> src/plot_*.py</tt> are saved.
