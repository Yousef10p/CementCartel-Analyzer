# Cement Cartel Analyzer

A Streamlit web app for analyzing whether cement price increases are justified by rising input costs or may indicate possible cartelization.

The app trains a regression model using cement industry data and predicts the expected cement sale price based on production input prices.



## Live Website

[Visit the Website Here](https://CementCartel.streamlit.app/)

## Project Idea

The cement industry may argue that sale prices increased because production costs increased.  
Consumers may argue that prices increased because of cartel behavior.

This app tests both arguments using a regression model:

CPRICE = f(FOIL, COAL, ELECTRICITY, GAS, PBAG, LIMESTONE, CLAY, GYPSUM, GDPGR)

## Features

- Loads cement industry data from `Assignment Data.xlsx`
- Trains a regression model automatically when the app starts
- Calculates:
  - Regression coefficients
  - P-values
  - R-squared
  - Number of observations
- Predicts the justified cement price based on user input
- Shows the contribution of each input variable
- Highlights statistically significant and insignificant variables
- Provides a simple cartel-risk interpretation based on explanatory power

## Dataset Variables

| Variable | Meaning |
|---|---|
| CPRICE | Cement sale price |
| FOIL | Furnace oil price |
| COAL | Coal price |
| ELECTRICITY | Electricity price |
| GAS | Gas price |
| PBAG | Price of packing bag |
| LIMESTONE | Limestone price |
| CLAY | Clay price |
| GYPSUM | Gypsum price |
| GDPGR | GDP growth rate |

## Econometric Model

The app uses multiple linear regression:

CPRICE = β0 + β1FOIL + β2COAL + β3ELECTRICITY + β4GAS + β5PBAG + β6LIMESTONE + β7CLAY + β8GYPSUM + β9GDPGR + ε

Where:

- CPRICE is the cement sale price
- Input prices are explanatory variables
- R-squared measures explanatory power
- P-values measure statistical significance

## Interpretation Logic

If production costs strongly explain cement prices:

- R-squared is high
- Key variables are statistically significant
- Coefficients are positive

This supports the producers’ argument.

If production costs weakly explain cement prices:

- R-squared is low
- Many variables are statistically insignificant
- Cement prices rise independently of costs

This may support cartelization concerns.

## Required File

The dataset file must exist in the same folder:

Assignment Data.xlsx

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- SciPy
- scikit-learn


