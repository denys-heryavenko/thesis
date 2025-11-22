# Unsupervised Entity Linking for Automated Metadata Generation in Austrian Government Job Descriptions Using NLP and the ESCO Classification System

This repository contains a reproducible pipeline to:
1. Collect data from job portal
2. Clean and segment job postings
3. Extract candidate phrases from the job postings
4. Embed phrases and units using Sentence-Transformers
5. Preprocess ESCO skills
6. Match jobs to ESCO **labels** (phrase-level, hybrid) and
**descriptions** (unit-level, semantic)
7. Write consolidated results to SQLite

All configuration is centralized in `config.py`, and all paths are
relative.

------------------------------------------------------------------------

## Repository Structure

- config.py
- scraper.py
- utils.py
- job_preprocessing.ipynb
- skills_preprocessing.ipynb
- matcher.ipynb
- data/
  - job_details.csv - input jobs file
  - skills_de.csv - ESCO input
  - v3.sqlite - generated SQLite DB
- requirements.txt
- README.md
- License

------------------------------------------------------------------------

## Scripts Overview

- `config.py` - Central configuration for paths, models, and parameters.
- `scraper.py`- Collects job postings from the Austrian public sector job portal: https://bund.jobboerse.gv.at/sap/bc/jobs/index.html
- `utils.py` - Cleaning, splitting, embeddings, and SQLite helpers.
- `job_preprocessing.ipynb` - Cleans, dedupes, splits, and embeds job postings.
- `skills_preprocessing.ipynb` - Cleans and embeds ESCO skills data
- `matcher.ipynb`- Matches job phrase/units to ESCO skills and writes results

------------------------------------------------------------------------

## Setup Instructions

### 1) Environment Setup

Open the project folder in Jupiter Lab and check if all files are present and if the directories work.

------------------------------------------------------------------------


### 1.1) Collect the data

If job data is missing, or reproducing it is required, the `scraper.py` script is a custom-built data collection tool that extracts
job postings from the Austrian public sector job portal. 

Run: `python scraper.py`
The output will be saved as `job_details.csv` in `/data`

------------------------------------------------------------------------



### 2) Add Data

Check if the raw job descriptions and esco concepts are in `/data`. Otherwise place the files:

-   `data/job_details.csv`- (must contain a text column, default: `aufgaben`)
-   `data/skills_de.csv` - ESCO German skills (with `preferredLabel`, `description`, etc)

Column names and paths can be adjusted in `config.py`.

------------------------------------------------------------------------

### 3) Configure Parameters

Open `config.py` and review:

-   **Paths**:`JOBS_INPUT_CSV`, `ESCO_CSV`, `SQLITE_PATH`
-   **Columns**: `COL_TEXT`, `COL_TITLE`, `ID_CANDIDATES`
-   **Models**: `SPACY_MODEL`, `KEYBERT_MODEL`, `EMBEDDER_MODEL`
-   **Phrase Extraction**: `KEYBERT_MAX_CANDIDATES`, `KEYBERT_SCORE_CUTOFF`, `ALLOWED_BIGRAMS`
-   **Matcher Settings**:
    -   `W_SEMANTIC`, `W_LEXICAL`, `W_CONTEXT`
    -   `TOP_K_PHRASE`, `MAX_UNIT_MATCHES`, `FINAL_TOP`
    -   `JOB_IDS` or `SAMPLE_JOBS` for testing subsets

------------------------------------------------------------------------

## Pipeline processes

### A) Job Preprocessing

Creates 3 tabels:

-   `jobs_clean` - cleaned text and deduped postings
-   `job_units` - bullet/sentence-level units + embeddings
-   `job_phrases` - extracted KeyBERT phrases + embeddings

Run: open the `job_preprocessing` notebook and run cell by cell.
Embeddings are computed using the model in `config.EMBEDDER_MODEL`.

------------------------------------------------------------------------

### B) ESCO Preprocessing

Creates 3 tables:

-   `esco_skills` (label/description/type)
-   `esco_labels` (label vectors)
-   `esco_desc` (description vectors)

Run: run the `skill_preprocessing` notebook cell by cell

------------------------------------------------------------------------

### C) Matching

Matches both **phrase-level** and **unit-level** similarities.

-   **Phrase-level hybrid**:`score = W_SEMANTIC * semantic + W_LEXICAL * lexical + W_CONTEXT * context`
-   **Unit-level**: cosine similarity between job units and ESCO  descriptions.

Creates `matched_results` with:

-   `job_id`, `title`, `text_deduped`
-   `phrase_matches_knowledge`, `phrase_matches_skills`
-   `unit_matches_knowledge`, `unit_matches_skills`
-   `final_meta` - merged, percentile-normalized top results

Run: run `matcher` notebook cell by cell

------------------------------------------------------------------------

## SQLite Tables

- `jobs_clean` - job_id, title, text_deduped, text_clean
- `job_units` - unit_text, embeddings
- `job_phrases` - KeyBERT phrases, POS tags, embeddings
- `esco_skills` - base ESCO table
- `esco_labels` - ESCO label embeddings
- `esco_desc` - ESCO description embeddings
- `matched_results` - all match results with JSON fields including an aggregated column `final_meta`

**Example of `final_meta` JSON:**

``` json
{
    "bucket": "knowledge",
    "char_span": {
        "end": 392,
        "start": 367
    },
    "context": "beratung studierenden",
    "label": "Beratung",
    "norm": 1.0,
    "score": 0.6750383377075195,
    "source": "phrase"
}
```

------------------------------------------------------------------------

## Citations

-   **ESCO** (European Skills, Competences, Qualifications and Occupations)
-   **Sentence-Transformers**: Reimers & Gurevych (2019),   *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks."*
-   **KeyBERT**: Grootendorst (2020), *"KeyBERT: Minimal keyword extraction with BERT."*

------------------------------------------------------------------------

## License

This project is licensed under the **MIT License** — see the [License](./License.txt) file for details.
