import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.expanduser(os.getenv("ENV_FILE", "~/.env")), override=True)

SEARCH_KEYWORDS = os.getenv("SEARCH_KEYWORDS", "software engineer")
SEARCH_LOCATION = os.getenv("SEARCH_LOCATION", "Remote")
RESULTS_WANTED = int(os.getenv("RESULTS_WANTED", "25"))
HOURS_OLD = int(os.getenv("HOURS_OLD", "72"))
