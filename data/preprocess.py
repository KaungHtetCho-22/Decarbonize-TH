import pandas as pd

def load_data(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df["Year"] = df["date"].dt.year
    df["Month"] = df["date"].dt.month
    df = df.sort_values("date")
    return df
