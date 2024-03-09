import streamlit as st
import numpy as np

# Install scikit-learn
st.write('Installing scikit-learn...')
import subprocess
subprocess.check_call(['pip', 'install', 'scikit-learn'])

# Now import LinearRegression from sklearn
from sklearn.linear_model import LinearRegression

# Generate some example data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title('Simple Linear Regression Model')

st.write('Enter a value for X to predict y:')
x_value = st.number_input('X')

# Predict function
def predict(x):
    return model.predict([[x]])[0][0]

if st.button('Predict'):
    y_pred = predict(x_value)
    st.write(f'Predicted y value: {y_pred:.2f}')
