import os

DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local").lower()

IS_LOCAL = DEPLOYMENT_MODE == "local"
IS_PRODUCTION = DEPLOYMENT_MODE == "production"

DATA_PATH = "data/raw/electronics_sample.jsonl"