import pandas as pd
from src.features import add_features

def test_add_features():
    df = pd.DataFrame({
        "close":[100,101,102,103,104,105,106,107,108,109],
        "open":[99,100,101,102,103,104,105,106,107,108],
        "high":[100,101,102,103,104,105,106,107,108,109],
        "low":[99,99,100,101,102,103,104,105,106,107],
    })
    res = add_features(df)
    # should create expected columns and no NaNs
    assert "rsi" in res.columns
    assert "target" in res.columns
    assert not res.isnull().any().any()
