import pandas as pd
from data.preprocessing import DataPreprocessor, load_feature_schema


def test_four_stages_and_missing_mask():
    schema = load_feature_schema("config/feature_schema.yaml")
    assert sorted(set({f["stage"] for f in schema})) == ["baseline", "intraoperative", "postoperative_24h", "preoperative"]
    df = pd.DataFrame({"age": [50, 60], "sex": ["M", None]})
    pre = DataPreprocessor(schema).fit(df)
    arr = pre.transform(df)
    idx = [f["name"] for f in schema].index("sex")
    assert arr["observed_mask"][0, idx] == 1
    assert arr["observed_mask"][1, idx] == 0
