#!/bin/bash

echo "Run tests? (y/n)"
read yntest
if [ "$yntest" == "y" ] # If y, run tests
then
  pipenv run pytest
fi

cd src
echo "Generate and preprocess Franke data? (y/n)"
read yn_generate_franke
if [ "$yn_generate_franke" == "y" ]
then
  echo "Generating Franke data with n_x = 20, n_y = 20 data points, sigma 1"
  pipenv run python generate_franke.py 20 20 1

  echo "Generating Franke data with n_x = 20, n_y = 20 data points, sigma 0.1"
  pipenv run python generate_franke.py 20 20 0.1

  echo "Generating Franke data with n_x = 200, n_y = 200 data points, sigma 0.1"
  pipenv run python generate_franke.py 200 200 0.1
fi


echo "Train Franke regression? (y/n)"
read yn_train_Franke
if [ "$yn_train_Franke" == "y" ]
then
  echo "Training neural network model and fitting hyperparameters for n_x = 20, n_y = 20 data points, sigma 1"
  pipenv run python train_nn_franke.py 20 20 1

  echo "Training neural network model and fitting hyperparameters for n_x = 20, n_y = 20 data points, sigma 0.1"
  pipenv run python train_nn_franke.py 20 20 0.1

  echo "Training neural network model and fitting hyperparameters for n_x = 200, n_y = 200 data points, sigma 0.1"
  pipenv run python train_nn_franke.py 200 200 0.1
fi


echo "Generate Franke plots? (y/n)"
read yn_plot_Franke
if [ "$yn_plot_Franke" == "y" ]
then
  echo "Plotting neural network model for n_x = 20, n_y = 20 data points, sigma 1"
  pipenv run python plot_regression.py 20 20 1

  echo "Plotting neural network model for n_x = 20, n_y = 20 data points, sigma 0.1"
  pipenv run python plot_regression.py 20 20 0.1

  echo "Plotting neural network model for n_x = 200, n_y = 200 data points, sigma 0.1"
  pipenv run python plot_regression.py 200 200 0.1
fi




echo "Preprocess credit card data? (y/n)"
read yn_generate_credit
if [ "$yn_generate_credit" == "y" ]
then
  echo "Creating scaled and one-hotted credit data design matrix"
  pipenv run python read_credit_data.py
fi

echo "Train logistic regression model on credit data? (y/n)"
read yn_train_logistic_credit
if [ "$yn_train_logistic_credit" == "y" ]
then
  echo "Running logistic regression, fitting hyperparameters"
  pipenv run python train_logreg_credit_data.py
fi

echo "Plot logistic regression results? (y/n)"
read yn_plot_logreg_credit
if [ "$yn_plot_logreg_credit" == "y" ]
then
  echo "Plotting logistic regression plots"
  pipenv run python plot_logreg_credit_data.py
fi

echo "Train neural network model on credit data? (y/n)"
read yn_train_nn_credit
if [ "$yn_train_nn_credit" == "y" ]
then
  echo "Running neural network classification, fitting hyperparameters"
  pipenv run python train_nn_credit_data.py
fi

echo "Plot neural network classification results? (y/n)"
read yn_plot_nn_credit
if [ "$yn_plot_nn_credit" == "y" ]
then
  echo "Plotting neural network classification plots"
  pipenv run python plot_nn_credit_data.py
fi

echo "Build report? (y/n)"
read ynreport
# If y, compile TeX document. The compilation is run many times because
# bibtex is usually non-cooperative...
if [ "$ynreport" == "y" ]
then
  cd ../doc/
  pdflatex -synctex=1 -interaction=nonstopmode report_2.tex
  bibtex report_2.aux
  pdflatex -synctex=1 -interaction=nonstopmode report_2.tex
  bibtex report_2.aux
  pdflatex -synctex=1 -interaction=nonstopmode report_2.tex
fi
