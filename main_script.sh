#!/bin/bash

echo "Run tests? (y/n)"
read yntest
if [ "$yntest" == "y" ] # If y, run tests
then
  pipenv run pytest
fi

cd src
echo "Generate Franke data? (y/n)"
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

echo "Preprocess credit card data? (y/n)"
read yn_generate_credit
if [ "$yn_generate_credit" == "y" ]
then
  echo "Creating scaled and one-hoted credit data design matrix"
  pipenv run python read_credit_dats.py
fi

echo "Train Franke regression? (y/n)"
read yn_train_Franke
if [ "$yn_train_Franke" == "y" ]
then
  echo "Training neural network model for n_x = 20, n_y = 20 data points, sigma 1"
  pipenv run python read_credit_dats.py 20 20 1

  echo "Training neural network model for n_x = 20, n_y = 20 data points, sigma 0.1"
  pipenv run python read_credit_dats.py 20 20 0.1

  echo "Training neural network model for n_x = 200, n_y = 200 data points, sigma 0.1"
  pipenv run python read_credit_dats.py 200 200 1
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
