import yaml

def test_locked_thresholds():
    with open("config/model_config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["thresholds"]["leakage_probability"] == 0.40
    assert cfg["thresholds"]["cr_popf_probability"] == 0.31
    assert cfg["model"]["num_time_stages"] == 4
