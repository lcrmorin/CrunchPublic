from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

def get_model_path(model_directory_path: str):
    model_path = os.path.join(model_directory_path, "model.joblib")
    return model_path

def train(X_train: pd.DataFrame,          
    y_train: pd.DataFrame,
    model_directory_path: str,
    target_column_names: [],
    prediction_column_names: [],
    feature_column_names: []) -> None:
    """
    Train here
    """

    model = MultiOutputRegressor(LinearRegression()).fit(X_train[feature_column_names], y_train[target_column_names])
    model_path = get_model_path(model_directory_path)
    joblib.dump(model, model_path)
    
def infer(
    X_test: pd.DataFrame,
    model_directory_path: str,
    id_column_name: str,
    moon_column_name: str,
    target_column_names: [],
    prediction_column_names: [],
    feature_column_names: []) -> pd.DataFrame:
    """
    Inference here
    """
    
    model_path = get_model_path(model_directory_path)
    model = joblib.load(model_path)
    
    ids = X_test[[moon_column_name, id_column_name]].copy()
    preds = pd.DataFrame(model.predict(X_test[feature_column_names]), columns=prediction_column_names)
        
    return pd.concat([ids, preds], axis=1)
