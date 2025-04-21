from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

dataset = pd.read_csv('clean.csv')
X = dataset.drop(columns=['co2'])  # replace 'target_column' with your actual target name
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)