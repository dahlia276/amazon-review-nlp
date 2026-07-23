from dotenv import load_dotenv
import os

load_dotenv()
# Production is the safe default for hosted deployments. Set
# DEPLOYMENT_MODE=local explicitly when running the full model comparison.
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "production").lower()

IS_LOCAL = DEPLOYMENT_MODE == "local"
IS_PRODUCTION = DEPLOYMENT_MODE == "production"

DATA_PATH = "data/raw/electronics_sample.jsonl"
