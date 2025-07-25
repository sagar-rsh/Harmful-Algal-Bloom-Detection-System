TIER_CONFIG = {
    "free": {
        "modalities": ['chlor_a'],
        "days": 5,
        "threads": 1,
        "model_path": "models/model_dt.pkl"
    },
    "tier2": {
        "modalities": ['chlor_a', 'Rrs_412', 'Rrs_443'],
        "days": 10,
        "threads": 4,
        "model_path": "models/model_lr.pkl"
    },
    "tier1": {
        "modalities": ['chlor_a', 'Rrs_412', 'Rrs_443'],
        "days": 10,
        "threads": 8, # Max threads
        "model_path": "models/model_rf.pkl"
    }
}