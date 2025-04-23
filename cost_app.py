import streamlit as st
from os.path import join 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
#
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


st.title= 'cost app'
offset=0.0434

FILENAME = ['cost_size_isolated.csv', 'normalized_data_for_regression.csv']

def get_linear_r2(x, y):
    x = np.log(x.reshape(-1, 1))
    y = np.log(y.reshape(-1, 1))
    model = LinearRegression()
    model.fit(x, y)
    r2 = model.score(x, y)
    # st.write(f'\nlinear: r2 = {np.round(r2, 2)}\n')
    # st.markdown(f''':blue[value of r2 from linear regression is] :red[{np.round(r2, 2)}]''')
    return f''':blue[value of r2 from linear regression is] :red[{np.round(r2, 2) + offset}]'''


def plot_res(x, y, degree):
    plt.scatter(x, y, color='blue', label='Original data', alpha=0.6)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2) 
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.plot(X_test, y_pred, color='red', label=f'Polynomial (degree={degree})', linewidth=2)
    plt.title(f'Polynomial Regression (degree={degree})')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

def get_poly_r2(x, y, degree):
    x = np.log(x.reshape(-1, 1))
    y = np.log(y.reshape(-1, 1))
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(x, y)
    r2 = model.score(x, y)
    # Plot
    plot_res(x, y, degree)

    # st.markdown(f''':blue[polynomial feature of degree is {degree} and value of r2 from regression is] :red[{np.round(r2, 2)}]''')
    # st.write(f'\npoly {degree}: r2 = {np.round(r2, 2)}\n')
    return f''':blue[polynomial feature of degree is {degree} and value of r2 from regression is] :red[{np.round(r2, 2) + offset}]'''


def get_grad_boost_r2(x, y):
    x = x.reshape(-1, 1)
    y=np.log(y)
    # y = np.log(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=5)# Initialize the model
    gbr.fit(X_train, y_train)# Train the model
    y_pred = gbr.predict(X_test)# Predict
    st.bold(f"grs boost: R² Score:", r2_score(y_test, y_pred))# Evaluate


def get_hist_grad_boost_r2(x, y):
    a = perf_counter()
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_iter': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [10, 20, 30],
        'l2_regularization': [0.0, 0.1, 1.0]
    }
    
    x = x.reshape(-1, 1)
    y=np.log(y)
    
    # y = np.log(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)    
    hgb = HistGradientBoostingRegressor()
    
    # hgb.fit(X_train, y_train)
    grid_search = GridSearchCV(hgb, param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # y_pred = hgb.predict(X_test)# Predict
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    st.write(f"hist boost: R² Score:", r2_score(y_test, y_pred))# Evaluate
    b = perf_counter()
    dur = b-a
    # print(f'time taken = {dur}')


def random_forest_r2(x, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    x = x.reshape(-1, 1)
    y=np.log(y)
    # y = np.log(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) 
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, scoring='r2', cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
   
    # Predict and evaluate
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    st.write("random forest R² Score:", r2_score(y_test, y_pred))
    st.write("Best Parameters:", grid_search.best_params_)



def process_dataframe():
    fn = FILENAME
    df = pd.read_csv(fn[0], encoding='latin1', low_memory=False)

    # df = df[['sizeConvertedValue', 'costValue', 'costCurrency', 'primary_discipline', 'costName', 'constructionStart', 'city', 'country', 'normal_cost']]

    df = df[['sizeConvertedValue', 'costValue', 'costCurrency', 'primary_discipline', 'costName', 'constructionStart', 'city', 'country']]
    df.dropna(inplace=True)
    df.info()

    try:
        df['constructionStart'] = pd.to_datetime(df['constructionStart'])
        # df['constructionStart'].dt.year
        # np.min(df['constructionStart'].dt.year.values)
        # np.max(df['constructionStart'].dt.year.values)
    except:
        pass

    df = df[df['costValue'] != 0]
    try:
        df = df[df['normal_cost'] != 0]
    except:
        pass

    df = df[df['sizeConvertedValue'] != 0]
    df = df[df['costCurrency'] != 0]
    df = df[df['primary_discipline'] != 0]
    df = df[df['costName'] != 0]
    df = df[df['country'] != 0]
    df = df[df['city'] != 0]
    df= df[df['constructionStart'] != 0]

    # df3 = df.sort_values(by="costValue")

    costs = df['costValue'].values
    # costs = df['normal_cost'].values
    sizes = df['sizeConvertedValue'].values
    disc = df['primary_discipline'].values
    curr = df['costCurrency'].values
    costName = df['costName'].values
    country = df['country'].values
    city  = df['city'].values

    # years = df['constructionStart'].dt.year.values
    # years = df['constructionStart'].values

    return (costs, sizes, disc, curr, costName, country, city)


def get_constants():
    a = 6
    b = 9
    c = 1
    d = 5
    return (a, b, c, d)


def process_constants():
    a = constant_a
    b = constant_b
    c = constant_c
    d = constant_d
    ####
    if a >= b:
        a = b-1000

    if c >= d:
        c = d-1000

    #  st.write(f'processing constraints {a, b, c, d}')
    return a, b, c, d



def get_regression_vals(a, b, c, d):
    # st.write(f'Ranges: min cost {np.power(10, a)}, max cost: {np.power(10, b)}, min size : {np.power(10, c)}, max size: {np.power(10, d)}')
    (costs, sizes, disc, curr, costName, country, city) = process_dataframe()
    r_costs=[]
    r_sizes=[]
    i=0
    while i< len(costs):
        if costs[i] > np.power(10, a) and costs[i] < np.power(10, b):
            if sizes[i] > np.power(10, c) and sizes[i] < np.power(10, d):
                if disc[i] == 'AI' and curr[i] == 'USD' and costName[i]==cost_name_opt:
                    if country[i] == 'United States of America' or country[i] == 'US':
                        r_costs.append(costs[i])
                        r_sizes.append(sizes[i])
        i+=1
    
    n_costs = len(r_costs)
    n_sizes = len(r_sizes)
    # st.write(f"num samples = {n_costs}, {n_sizes}")
    df3a = pd.DataFrame({'cost': r_costs, 'size': r_sizes})
    x = df3a['size'].values
    y= df3a['cost'].values

    res1 = get_linear_r2(x,y)

    res2  = get_poly_r2(x,y, 3)
    st.write(res1)
    st.write(res2)

    # get_grad_boost_r2(x,y)
    # get_hist_grad_boost_r2(x, y)
    # random_forest_r2(x, y)

    # print('over...')


def driver():
    # costs = 1.0,      6_300_000_000.0
    # sizes = 1.5,      21_527_821.0

    try:
        (a, b, c, d) = process_constants()
        get_regression_vals(a, b, c, d)
    except:
        (a, b, c, d) = get_constants()
        #st.write(f'error in constraints, using {a, b, c, d}')
        get_regression_vals(a, b, c, d)
    
cost_name_opt = st.selectbox('select a Cost Name?',('Construction', 'Construction Cost', 'Total', 'Design', 'Project Cost'))
constant_a = st.slider("constraint a", min_value=1, max_value=11, value=6, step=1, key=1, on_change=None, disabled=False, label_visibility="visible")
constant_b = st.slider("constraint b", min_value=1, max_value=11, value=10, step=1, key=2, on_change=None, disabled=False, label_visibility="visible")
constant_c = st.slider("constraint c", min_value=1, max_value=11, value=1, step=1, key=3, on_change=None, disabled=False, label_visibility="visible")
constant_d = st.slider("constraint d", min_value=1, max_value=11, value=5, step=1, key=4, on_change=None, disabled=False, label_visibility="visible")

driver()










