def prepare_features(df):
    # Use historical values as features (lags)
    for lag in range(1, 13):
        df[f"lag_{lag}"] = df["co2_emission"].shift(lag)
    
    df = df.dropna().reset_index(drop=True)
    X = df[["Year", "Month"] + [f"lag_{i}" for i in range(1, 13)]]
    y = df["co2_emission"]
    return X, y
