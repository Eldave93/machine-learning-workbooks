def create_example_model(df_red_, output):

    import pandas as pd
    from imblearn.pipeline import Pipeline
    from imblearn import FunctionSampler
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression

    # remove duplicated values
    def drop_duplicated(X,y):
        df = pd.concat([X,y], axis=1)
        df = df.drop_duplicates()
        return df.iloc[:,:-1], df.iloc[:,-1]

    DD =  FunctionSampler(func=drop_duplicated,
                          validate=False)

    # standardises all variables
    scaler = StandardScaler() 

    # here is the model we want to use.
    reg = LinearRegression()

    # create our pipeline for the data to go through.
    # This is a list of tuples with a name (useful later) and the function.
    reg_pipe = Pipeline([
        ("drop_duplicated", DD),
        ("scaler", scaler),
        ("model", reg)
    ])

    y_train = df_red_.loc[:,output]
    X_train = df_red_.drop(output, axis=1)

    return reg_pipe.fit(X_train, y_train)

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, method='pearson', threshold=0.8): 
        
        """
        data_x: features to check for correlations
        method: how to calculate the correlation
        threshold: the cutoff to remove one of the correlated features
        """
        
        
        self.method = method
        self.threshold = threshold
        
    def fit(self, X, y=None):
        # make a copy
        X_ = X.copy()

        # turn to dataframe if a numpy array
        if isinstance(X_, (np.ndarray, np.generic)):
            X_ = pd.DataFrame(X_)

        # from https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
        cor_matrix  = X_.corr(method=self.method).abs() # get correlation and remove -
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.threshold)]
        
        # get the names of features that are corrleated
        correlated = []
        for column in upper_tri.columns:
            if any(upper_tri[column] > self.threshold):
                corr_data = upper_tri[column][upper_tri[column] > self.threshold]
                for index_name in corr_data.index:
                    correlated.append((index_name, corr_data.name))
        
        self.features_to_drop_ = np.array(to_drop)
        self.correlated_feature_sets_ = np.array(correlated)
        
        return self
        
    def transform(self, X, y=None):
        X_ = X.copy()
        
        # turn to dataframe if a numpy array
        if isinstance(X_, (np.ndarray, np.generic)):
            X_ = pd.DataFrame(X_)
        
        return X_.drop(self.features_to_drop_, axis=1)

def bike_data_prep(data):
    import pandas as pd

    data_ = data.copy()
    
    to_drop = ['casual', 'registered']
    if any(x in to_drop for x in data_.columns):
        data_.drop(['casual', 'registered'], axis=1, inplace=True)
        
    data_.rename(columns={'count':'riders', 'atemp': 'realfeel'}, inplace=True)
    data_['datetime'] = pd.to_datetime(data_['datetime'], format='%Y-%m-%d %H:%M:%S')
    data_.set_index('datetime', inplace=True)
    
    return data_


def create_example_bike_model(data_, output):

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    def extract_hour(dt):
        return dt.hour
    
    def create_hour_feat(data):
        data_ = data.copy()
        data_['hour'] = data_.index.map(extract_hour)
        return data_
    
    data_ = bike_data_prep(data_)
    
    # make compatible with a scikit-learn pipeline
    hour_feat_func = FunctionTransformer(func=create_hour_feat,    # our custom function
                                         validate=False)           # prevents input being changed to numpy arrays
    
    
    hour_onehot = ColumnTransformer(
        # apply the `OneHotEncoder` to the "hour" column
        [("OHE", OneHotEncoder(drop="first"), ["hour"])],
        # don't touch all other columns, instead concatenate it on the end of the
        # changed data.
        remainder="passthrough"
    ) 
    
    lin_dummy_pipe = Pipeline([
        ("create_hour", hour_feat_func),
        ("encode_hr", hour_onehot),
        ("model", LinearRegression())
    ])

    return lin_dummy_pipe.fit(data_.drop(["temp", "riders"], axis=1), data_.loc[:,"riders"])


def example_residual_plot(dataframe, x_feat, y_feat):
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import matplotlib.pyplot as plt

    # fit simple linear regression model
    linear_model = ols(y_feat+' ~ '+x_feat,
                       data=dataframe).fit()

    # creating regression plots
    fig = sm.graphics.plot_regress_exog(linear_model,
                                        x_feat)

    # change the plot to the plotted and fitted 
    fig.axes[0].clear()
    fig.axes[0].scatter(dataframe[x_feat], dataframe[y_feat])
    fig.axes[0].set_xlabel(x_feat)
    fig.axes[0].set_ylabel(y_feat, labelpad=15)
    fig.axes[0].set_title(x_feat + ' versus ' + y_feat, {'fontsize': 15})
    sm.graphics.abline_plot(model_results=linear_model, ax = fig.axes[0], c="black")

    # remove unneccisary plots
    fig.delaxes(fig.axes[2])
    fig.delaxes(fig.axes[2])

    # tidy label
    fig.axes[1].set_title('residuals versus '+y_feat, {'fontsize': 15})

    # residual lines on left plot
    for i, num in enumerate(dataframe[y_feat]):
        value = dataframe[y_feat][i]
        prediction = linear_model.predict(dataframe[x_feat][i:i+1])[0]
        if value > prediction:
            fig.axes[0].vlines(dataframe[x_feat][i], ymin=prediction, ymax=value, color="red", linestyles="dashed")
        else:
            fig.axes[0].vlines(dataframe[x_feat][i], ymin=value, ymax=prediction, color="red", linestyles="dashed")

    # residual lines on right plot    
    for i, num in enumerate(dataframe[y_feat]):
        resid = linear_model.resid[i]
        if 0 > resid:
            fig.axes[1].vlines(dataframe[x_feat][i], ymin=resid, ymax=0, color="red", linestyles="dashed")
        else:
            fig.axes[1].vlines(dataframe[x_feat][i], ymin=0, ymax=resid, color="red", linestyles="dashed")

    plt.show()


def plot_cv_indices(cv, X, ax, n_splits, lw=10):
    # edited from https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py

    """Create a sample plot for indices of a cross-validation object."""
    from matplotlib.patches import Patch
    import matplotlib.pyplot as plt
    import numpy as np

    cmap_cv = plt.cm.coolwarm
    cmap_data = plt.cm.Paired

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            X.index,
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )

    # Plot the data at the top
    ax.scatter(
        X.index, [ii + 1.5] * len(X), marker="_", lw=lw, cmap=cmap_data
    )

    # Formatting
    yticklabels = list(range(n_splits))+ ["data"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        xlabel="Sample index",
        ylabel="CV iteration",
    )
    plt.xticks(rotation=45, ha='right')
    ax.set_title("{}".format(type(cv).__name__), fontsize=15)
    ax.legend(
        [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.02))],
        ["Testing set", "Training set"],
        loc=(1.02, 0.8),
    )
    return ax


# from Colin Rundel (http://www2.stat.duke.edu/~cr173/)
def get_coefs(m):
    """Returns the model coefficients from a Scikit-learn model object as an array,
    includes the intercept if available.
    """
    import sklearn
    import numpy as np
    
    
    # If pipeline, use the last step as the model
    if (isinstance(m, sklearn.pipeline.Pipeline)):
        m = m.steps[-1][1]
    
    
    if m.intercept_ is None:
        return m.coef_
    
    return np.concatenate([[m.intercept_], m.coef_])

# from Colin Rundel (http://www2.stat.duke.edu/~cr173/)
def model_fit(m, X, y, plot = False):
    """Returns the root mean squared error of a fitted model based on provided X and y values.
    
    Args:
        m: sklearn model object
        X: model matrix to use for prediction
        y: outcome vector to use to calculating rmse and residuals
        plot: boolean value, should fit plots be shown 
    """
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    
    y_hat = m.predict(X)
    rmse = mean_squared_error(y, y_hat, squared=False)
    
    res = pd.DataFrame(
        data = {'y': y, 'y_hat': y_hat, 'resid': y - y_hat}
    )
    
    if plot:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(121)
        sns.lineplot(x='y', y='y_hat', color="grey", data =  pd.DataFrame(data={'y': [min(y),max(y)], 'y_hat': [min(y),max(y)]}))
        sns.scatterplot(x='y', y='y_hat', data=res).set_title("Fit plot")
        
        plt.subplot(122)
        sns.scatterplot(x='y', y='resid', data=res).set_title("Residual plot")
        plt.hlines(y=0, xmin=np.min(y), xmax=np.max(y), linestyles='dashed', alpha=0.3, colors="black")
        
        plt.subplots_adjust(left=0.0)
        
        plt.suptitle("Model rmse = " + str(round(rmse, 4)), fontsize=16)
        plt.show()
    
    return rmse

# from Colin Rundel (http://www2.stat.duke.edu/~cr173/)
# TODO: Could get this working with hour?
def ridge_coef_alpha_plot(X_train, X_val, y_train, y_val):
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    def extract_hour(dt):
        return dt.hour
    
    def create_hour_feat(data):
        data_ = data.copy()
        data_['hour'] = data_.index.map(extract_hour)
        return data_
    
    #X_train = data_prep(X_train)
    #X_val = data_prep(X_val)
    
    # make compatible with a scikit-learn pipeline
    hour_feat_func = FunctionTransformer(func=create_hour_feat,    # our custom function
                                         validate=False)           # prevents input being changed to numpy arrays
    
    
    hour_onehot = ColumnTransformer(
        # apply the `OneHotEncoder` to the "hour" column
        [("OHE", OneHotEncoder(drop="first"), ["hour"])],
        # don't touch all other columns, instead concatenate it on the end of the
        # changed data.
        remainder="passthrough"
    ) 

    scaler = ColumnTransformer(
        # apply the `StandardScaler` to the numerical data
        [("SS", StandardScaler(), ["realfeel", "humidity", "windspeed"])],
        # don't touch all other columns, instead concatenate it on the end of the
        # changed data.
        remainder="passthrough"
    ) 
    
    alphas = np.logspace(-2, 3, num=200)

    betas = [] # Store coefficients
    rmses = [] # Store validation rmses

    col_names = ['season', 'holiday', 'workingday', 'weather', 'realfeel', 'humidity', 'windspeed', 'hour']

    for a in alphas:
        m = Pipeline([
            #("create_hour", hour_feat_func),
            #("pandarizer",FunctionTransformer(lambda x: pd.DataFrame(x, columns = col_names))),
            ("scaler", scaler),
            #("pandarizer2",FunctionTransformer(lambda x: pd.DataFrame(x, columns = col_names))),
            #("encode_hr", hour_onehot),
            ("model", LinearRegression())
        ]).fit(X_train, y_train)

        # We drop the intercept as it is not included in Ridge's l2 penalty and hence not shrunk
        betas.append(get_coefs(m)[1:]) 
        rmses.append(model_fit(m, X_val, y_val))

    res = pd.DataFrame(
        data = betas,
        columns = X_train.columns # Label columns w/ feature names
    ).assign(
        alpha = alphas,
        rmse = rmses
    ).melt(
        id_vars = ('alpha', 'rmse')
    )

    sns.lineplot(x='alpha', y='value', hue='variable', data=res).set_title("Coefficients")
    plt.show()

# from Colin Rundel (http://www2.stat.duke.edu/~cr173/)
def lasso_coef_alpha_plot(X_train, X_val, y_train, y_val):
    import numpy as np
    from sklearn.pipeline import Pipeline
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Lasso

    alphas = np.logspace(-2, 2, num=200)

    betas = [] # Store coefficients
    rmses = [] # Store validation rmses

    scaler = ColumnTransformer(
        # apply the `StandardScaler` to the numerical data
        [("SS", StandardScaler(), ["realfeel", "humidity", "windspeed"])],
        # don't touch all other columns, instead concatenate it on the end of the
        # changed data.
        remainder="passthrough"
    ) 

    for a in alphas:
        m = Pipeline([
            ("SS", scaler),
            ("Ridge", Lasso(alpha=a))]
        ).fit(X_train, y_train)
        
        # We drop the intercept as it is not included in Ridge's l2 penalty and hence not shrunk
        betas.append(get_coefs(m)[1:]) 
        rmses.append(model_fit(m, X_val, y_val))
        
    res = pd.DataFrame(
        data = betas,
        columns = X_train.columns # Label columns w/ feature names
    ).assign(
        alpha = alphas,
        rmse = rmses
    ).melt(
        id_vars = ('alpha', 'rmse')
    )

    sns.lineplot(x='alpha', y='value', hue='variable', data=res).set_title("Coefficients")
    plt.show()


def clean_ufos(ufo):
    import pandas as pd
    ufo.Time = pd.to_datetime(ufo.Time, format='%m/%d/%Y %H:%M')
    ufo.set_index('Time', inplace=True)
    
    return ufo


def tidy_eu_passengers(data):
    import pycountry
    import pandas as pd
    from re import match

    # ------------
    # TIDY COLUMNS
    # ------------

    # rename columns
    airlines = data.rename({
        "geo\\time":"country",
        "tra_meas":"measurement",
        "tra_cov": "coverage"
    }, axis="columns").copy()

    non_date_cols = list(airlines.columns[0:5])

    # remove the space in column names
    airlines.columns = airlines.columns.str.replace(' ', '')

    # just get the month columns
    filtered_values = list(filter(lambda v: match('\d+M\d+', v), airlines.columns))

    # reduce columns down to years with months
    airlines = airlines[non_date_cols+filtered_values]

    # make a date column
    airlines = pd.melt(airlines,
                       id_vars=non_date_cols,
                       var_name="date",
                       value_name='vals') 

    # ---------
    # TIDY DATA
    # ---------

    # replace the 'M' with a dash
    airlines.date = airlines.date.str.replace('M', '-')

    # change to a datetime
    airlines.date = pd.to_datetime(airlines.date, format='%Y-%m')

    #set the date as the index
    airlines.set_index('date', inplace=True)

    # get a dictionary with the codes and the country name
    country_dict = {}
    for country in airlines["country"].unique():
        try:
            country_dict[country] = pycountry.countries.lookup(country).name
        except:
            pass

    # use the dictionary to replace the codes
    airlines.country = airlines.country.replace(country_dict)
    # change ":" to nan
    airlines = airlines.replace(": ", np.nan)

    # change the values to float
    airlines.vals = airlines.vals.astype("float", errors='ignore')

    # sort earliest to most recent
    airlines.sort_index(inplace=True)

    return airlines
    