# Acronime specifice Capitolului 13 SFM (Învățare automată / Machine learning)
# format: acronim -> (forma de origine, limba de origine, traducere RO, traducere EN)
#   ex.: 'EVT': ('Extreme Value Theory', 'en', 'teoria valorilor extreme', None)
# OVERRIDE_CH (optional): acronim -> tuplu, sens diferit doar in acest capitol.
# GBM (Geometric Brownian Motion in dictionarul comun) nu este folosit aici: gradient boosting apare ca GB.
EXTRA = {
    'SSE': ('Sum of Squared Errors', 'en', 'suma pătratelor erorilor', None),
    'RBF': ('Radial Basis Function (network)', 'en', 'rețea cu funcții de bază radiale', None),
    'NN-ARCH': ('Neural-Network ARCH', 'en', 'model ARCH cu rețea neuronală', None),
    'LRV': ('Long-Run Variance', 'en', 'varianța pe termen lung', None),
    'QR': ('Quantile Regression', 'en', 'regresie cuantilică', None),
    'QGB': ('Quantile Gradient Boosting', 'en', 'gradient boosting cu pierdere cuantilică', None),
    'OLS-3': ('OLS with 3 predictors (size, book-to-market, momentum)', 'en', 'OLS cu 3 predictori (mărime, raportul valoare contabilă/valoare de piață, momentum)', None),
    'NN': ('Neural Network (NN$k$: $k$ hidden layers)', 'en', 'rețea neuronală (NN$k$: $k$ straturi ascunse)', None),
    'NN1': ('Neural Network with 1 hidden layer', 'en', 'rețea neuronală cu un strat ascuns', None),
    'NN3': ('Neural Network with 3 hidden layers', 'en', 'rețea neuronală cu 3 straturi ascunse', None),
    'NN4': ('Neural Network with 4 hidden layers', 'en', 'rețea neuronală cu 4 straturi ascunse', None),
    'NN5': ('Neural Network with 5 hidden layers', 'en', 'rețea neuronală cu 5 straturi ascunse', None),
    'OOS': ('Out-Of-Sample', 'en', 'în afara eșantionului', None),
    'SPX': ('S&P 500 index (ticker symbol)', 'en', 'indicele S&P 500 (simbolul de tranzacționare)', None),
}
OVERRIDE_CH = {
    'RF': ('Random Forest', 'en', 'pădure aleatoare', None),
}
