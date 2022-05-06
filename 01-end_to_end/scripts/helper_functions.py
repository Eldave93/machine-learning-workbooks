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