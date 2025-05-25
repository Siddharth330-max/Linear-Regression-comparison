### Implementing Linear Regression from scratch and comparing it with sklearn.LinearRegression  

This project compares two implementations of Linear Regression:
1. A custom implementation using only NumPy
2. A standard implementation using `sklearn.linear_model.LinearRegression`   


## Files
1. 'EDA.ipynb': Cotains basic Data Analysis of the dataset
2. 'LinearRegression.py' : Defines the functions regarding custom Linear Regression
3. `train_custom.py`: Linear Regression from scratch
4. `train_sklearn.py`: Using `sklearn`
5. `Weight-Height for linear regression.csv`: Dataset used    

##  Mean Squared Error Comparison

| Model          | Training MSE | Test MSE     |
|----------------|--------------|--------------|
| Custom         | 0.18807      | 0.05060      |
| scikit-learn   | 0.18802      | 0.04960      |   

## Observations

- Both models perform similarly, with scikit-learn having a slightly lower test MSE.
- Visualizations confirm that the best-fit lines are almost identical.
- Standardization significantly improves model convergence for the custom model.
- This demonstrates that even with a simple gradient descent, we can closely match sklearn's results   

## How to run

---cmd  
python train_custom.py
python train_sklearn.py