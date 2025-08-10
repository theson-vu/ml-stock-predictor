import pandas as pd
import src.data_fetch as data_fetch

def test_fetch_bars_monkeypatched():
    # Create fake bars data
    fake_df = pd.DataFrame({
        "open": [1, 2],
        "high": [1.5, 2.5],
        "low": [0.5, 1.5],
        "close": [1.2, 2.2],
        "volume": [100, 200]
    })

    class DummyClient:
        def get_bars(self, symbol, timeframe, start=None, end=None):
            class Wrapper:
                df = fake_df  # assign directly here
            return Wrapper()

    client = DummyClient()
    df = data_fetch.fetch_bars("FAKE", "2020-01-01", "2020-01-03", client=client)

    assert not df.empty
    assert "open" in df.columns
