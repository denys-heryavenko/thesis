#Configurations and Constants
#This file contains a collection of fixed variables, models, and various elements used in the dual-approach EL pipeline. The main purpose of it is to enable a simple and central overview of the parameters that directly influence the pipeline's outputs and support experimentation with different variables.


### Directory management

from pathlib import Path
REPO_ROOT = Path().resolve() 
DATA_DIR = REPO_ROOT / "data"

JOBS_INPUT_CSV = DATA_DIR / "job_details.csv" #saved by the scraper - raw job data
ESCO_CSV = DATA_DIR / "skills_de.csv" #raw ESCO dataset
SQLITE_PATH = DATA_DIR / "db.sqlite" #central database

### Jobs Scraper Settings

SCRAPER_URL = "https://bund.jobboerse.gv.at/sap/bc/jobs/index.html#"
SCRAPER_WAIT = 30
SCRAPER_HEADLESS = False #we want to see what the scraper is doing
SCRAPER_RENDER_DELAY = 1
SCRAPER_AFTER_CLICK_DELAY = 0.25
SCRAPER_CSV_OUT = DATA_DIR / "job_details.csv" #raw data export

### Column Names and ID detection

COL_TEXT = "aufgaben"
COL_TITLE = "title"
ID_CANDIDATES = ["job_id", "id", "JobID", "jobId", "stellen_id", "anzeige_id"] #in case the dataset already has a primary key (e.g., ID)

### NLP Models

SPACY_MODEL = "de_core_news_sm" #main german spaCy model
FALLBACK_SPACY_MODEL = "xx_sent_ud_sm" #just in case if the main one doesn't load
KEYBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2" 
EMBEDDER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

### Extra Stopwords

EXTRA_STOPWORDS_DE = {"sowie", "bzw", "insbesondere", "aufgaben", "anforderungen",
                      "bewerbung", "bewerben", "team", "arbeit", "tätigkeit", "taetigkeit",
                      "profil", "wir", "sie", "uns", "ihre", "unser", "vollzeit","teilzeit", "mwd", "m w d"}

### Text Splitting config

MIN_WORDS_PER_UNIT = 3  #minimum length for each cleaned unit

### KeyBERT Settings

KEYBERT_MAX_CANDIDATES = 300
KEYBERT_SCORE_CUTOFF = 0.3
ALLOWED_BIGRAMS = [("ADJ", "NOUN"),("NOUN", "NOUN"),("NOUN", "VERB"),("NOUN", "ADJ"),("VERB", "NOUN"),]

### Embedding config

EMBED_BATCH_SIZE = 256

### Matching / Ranking Configuration

JOB_IDS = [] #one can process concrete jobs, to examine them e.g., [39, 190, 310]
SAMPLE_JOBS = None # here a random number of jobs can be selected and processed. Used for testing and evaluation

# ranking caps
TOP_K_PHRASE = 20 #top-k matches per skillType on phrase level. Number per job = *2
MAX_UNIT_MATCHES = 20 #same, but for units
FINAL_TOP = 40

#hybrid weighting for phrase-level scoring
W_SEMANTIC = 0.6
W_LEXICAL  = 0.25
W_CONTEXT  = 0.15

