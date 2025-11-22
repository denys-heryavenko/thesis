import json
import sqlite3

from functools import lru_cache #the language models are cached to increase efficiency 
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

import numpy as np
import pandas as pd
import regex as re
import spacy

import config as C #config file


### Regex setup for text cleaning
#General regex expressions are defined to be used in text normalization. Specifically focused on maintaining structure (bullets, dashes) so the unit extraction works.

BAD_CHARS = re.compile(r"[^\p{L}\p{M}\p{N}\s\.\,\;\:\-\(\)\/\+\*\•\—\–\·]") #remove everything except normal punctuation + bullets/dashes
WS = re.compile(r"\s+") #all whitespaces are collapsed here. This is relevant for phrase-level cleaning, when structure doesn't matter
WS_NO_NL = re.compile(r"[ \t\f\v]+") # opposite to the previous one: we collapse spaces/tabs, but not newlines, to maintain structure (e.g., bullets...)
NL_TRIM = re.compile(r"[ \t\f\v]*\n[ \t\f\v]*") #triming spaces around newlines
NL_COLLAPSE = re.compile(r"(?:\n\s*){2,}") #remove multiple blank lines


### Text normalization and deduplication
#In the original job dataset, the aufgaben text was duplicated, so this required an additional function that records each line and checks if it already exists in the aufgaben text.

def dedupe_lines(text):
    
    if not isinstance(text, str) or text.strip() == "":
        return ""
        
    seen = set()
    result = []

    lines = text.splitlines()
    
    for line in lines:
        clean_line = line.strip()
        
        if clean_line == "":
            continue
            
        check = clean_line.lower() #lowercase it to compare to the lines in the set
 
        if check not in seen:
            seen.add(check)
            result.append(clean_line)
            
    return "\n".join(result) #rejoin the unique lines back to a single string

def clean_text(text):
   
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n") #to be safe, different types of newlines are converted into standardized ones
    
    text = BAD_CHARS.sub(" ", text) #remove unwanted characters
    text = NL_TRIM.sub("\n", text)
    text = WS_NO_NL.sub(" ", text) #NOT WS, because we want to preserve structure for unit level extraction
    text = NL_COLLAPSE.sub("\n", text)

    return text.strip()

#tokenizer helper functions used in matcher
TOKEN_SPLIT = re.compile(r"[^\p{L}\p{M}\p{N}\+]+")

def tokenize(text, stopwords=None, min_len=2):
   
    if not isinstance(text, str):
        return []
        
    text = text.lower()
    parts = TOKEN_SPLIT.split(text)
    tokens = []

    for t in parts:
        if t == "":
            continue
        if stopwords is not None:
            if t in stopwords:
                continue
        if min_len is not None:
            if len(t) < min_len:
                continue
        tokens.append(t)

    return tokens


### Cached model loaders
#Language models, transformers, and additional tools are loaded once and cached so they can be used throughout the pipeline without the need to always load again. 

@lru_cache(maxsize=1)
def get_spacy(lang=C.SPACY_MODEL):
    
    #the main model is loaded, if unsuccessful it falls back to a simple model to not crash the whole process. 
    try:
        nlp = spacy.load(lang, disable=["ner"]) #phrase extraction is used in the pipeline, so NER is not needed (it's quite heavy)
    except Exception:
        nlp = spacy.load(C.FALLBACK_SPACY_MODEL, disable=["ner"]) 

    
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names: #some spaCy models might not have sentence splitters. In that case a simple rule-based splitter is used.
        nlp.add_pipe("sentencizer")

    return nlp

@lru_cache(maxsize=1)
def get_keybert(model=C.KEYBERT_MODEL):
    return KeyBERT(model)

@lru_cache(maxsize=1)
def get_embedder(model=C.EMBEDDER_MODEL):
    return SentenceTransformer(model)


@lru_cache(maxsize=1)
#here the base stopwords are loaded, but also enriched with variations and the curated list
def get_stopwords_de():
    
    nlp = get_spacy()
    base_words = set()

    for w in nlp.Defaults.stop_words:
        w = w.lower()
        base_words.add(w)
    
    base_words.update(C.EXTRA_STOPWORDS_DE) #add the curated list

    #add variations of spelling
    def make_variants(word):
        variants = set()
        variants.add(word)
        variants.add(word.replace("ß", "ss"))
        variants.add(word.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue"))
        variants.add(word.replace("ae", "ä").replace("oe", "ö").replace("ue", "ü"))
        return variants

    all_stopwords = set() #final set of stopwords
    for word in base_words:
        all_stopwords.update(make_variants(word))

    return sorted(all_stopwords)


@lru_cache(maxsize=1)
def get_vectorizer(): #vectorizer for KeyBERT
   
    stopwords = get_stopwords_de()

    vectorizer = CountVectorizer(
        ngram_range=(1, 3),        
        stop_words=stopwords, #we exclude stopwords
        token_pattern=r"(?u)\b\w[\w\-]{1,}\b",  #what a token should be
        strip_accents=None, #we don't remove ä, ö, ü...
        dtype=np.float32  #store in 32-bit floats for less memory usage
    )

    return vectorizer


### Sentence level extraction / unit splitting
# Here, the main functions are defined that support the splitting of sentences into individual units. The process breaks down into 3 situations:

# 1) If there is a bullet structure - split by bullets
# 2) Newlines structure - split by newlines
# 3) And if it's a paragraph, use spaCy sentence splitter


#here we check if the sentence starts with a bullet or any other order identifier (1.,(2), a...)
LEADING_BULLET = re.compile(
    r"""^\s*(?:                              
            [•\-\–\—\*·]                      
          | (?:\d+[\.\)])                     
          | \([a-zA-Z]\)                      
          | (?:[a-zA-Z]\))                    
        )\s+""",
    re.X
)
RULE_LINE = re.compile(r"^\s*[–—\-]{3,}\s*$")#if there are separators in text made from dashes, we want to identify them

#helper functions used later
def is_bullet_line(text):
    return bool(LEADING_BULLET.match(text)) #return true if line starts with a bullet sign

def strip_bullet(text):
    cleaned = LEADING_BULLET.sub("", text) #remove the detected bullet symbol
    return cleaned.strip()

def word_count(text):
    words = text.split()
    return len(words)

#splitting bullet-structure into units
def split_by_bullets(text):
    
    units = []
    current = []
    
    lines = text.splitlines()
    
    for line in lines:
        line = line.rstrip()   #remove spaces at the end

        #if the line is empty or just dashes, we close current and move on
        if line == "" or RULE_LINE.match(line):
            if current:
                units.append(" ".join(current).strip())
                current = []
            continue

        
        if is_bullet_line(line):
            if current:
                units.append(" ".join(current).strip()) #if inside bullet,close it
            current = [strip_bullet(line)] #add the bullet part without the symbol

        else:
            if current:
                current.append(line.strip())
            else:
                units.append(line.strip())

    if current:
        units.append(" ".join(current).strip()) #double check if something is left in current. add it

    cleaned_units = []
    for u in units: #in case there are empty entries for any reason, remove them
        if u:
            cleaned_units.append(u)

    return cleaned_units


#The next function splits the job description text into separate units. The idea is as follows: 
# 1) If any bullets are detected at line starts - use bullet splitting
# 2) Else, if there are newlines - treat each non-empty line as its own unit
# 3) Else - treat as a paragraph and apply spaCy sentence segmentation, merging short sentences.

def split_units(text, nlp, min_words_per_unit=C.MIN_WORDS_PER_UNIT):
    
    if not text:
        return []

    #we check if there is at least one bullet (case 1)
    has_bullet = False
    for line in text.splitlines():
        if is_bullet_line(line):
            has_bullet = True
            break

    if has_bullet:
        return split_by_bullets(text)

    #if no bullets we check for newlines (case 2)
    if "\n" in text:
        raw_lines = text.splitlines() #non-empty lines
        cleaned_lines = []

        for line in raw_lines:
            
            line = line.strip()
            if line == "":
                continue
            if RULE_LINE.match(line):
                continue
          
            line = strip_bullet(line) #just in case there is a bullet somewhere
            if line:
                cleaned_lines.append(line)

        return cleaned_lines

    #if it's a single paragraph then we use spaCy to segment by sentences (case 3)
    doc = nlp(text)
    sentences = []

    for sent in doc.sents:
        s = sent.text.strip()
        if s:
            sentences.append(s)

    merged = []

    for seg in sentences:
        if not seg:
            continue

        if len(merged) == 0:
            merged.append(seg)
        else:
            #we check the word count to see if we need to merge 
            last = merged[-1]
            if word_count(seg) < min_words_per_unit or word_count(last) < min_words_per_unit:
                merged[-1] = (last + " " + seg).strip()
            else:
                merged.append(seg)

    final_units = []
    for m in merged:
        if m:
            final_units.append(m)

    return final_units


### Phrase-level extraction
# The extraction process on phrase-level includes a couple of helper functions and the main extractor. 
# After KeyBERT extracted the candidate phrases, they are put through the following filtering steps:
# 1) minimum KeyBERT score
# 2) normalization using tidy_phrase()
# 3) reduction to bigrams
# 4) filter by allowed POS patterns
# 5) deduplication by lemma (keep highest-scoring option)
# 6) overlap pruning at surface level

ARTICLE_EDGES = re.compile(r"^(?:der|die|das|ein|eine|einen|einem|einer|n|für|fur)\s+|\s+(?:der|die|das|ein|eine|für|fur|n)$",re.I)
# is used to clean extracted phrases before POS filtering (clean the edges of a phrase)

def tidy_phrase(text): # helper function that cleans the KeyBERT extracted phrase

    text = text.strip()
    text = ARTICLE_EDGES.sub("", text)
    text = WS.sub(" ", text) #multiple spaces and tabs are replaced here with on space
    text = text.lower()

    return text

ALLOWED_BIGRAMS = set()
#convert the POS bigrams into a set of tuples
for item in C.ALLOWED_BIGRAMS:
    bigram = tuple(item)
    ALLOWED_BIGRAMS.add(bigram)

def content_tokens(doc):

    content = []
    #we remove stopwords, punctuation and spaces
    for t in doc:
        if t.is_stop:
            continue
        if t.is_punct:
            continue
        if t.is_space:
            continue
        content.append(t)
    return content


def lemma_bigram(doc): #return bigrams and their versions
    
    tokens = content_tokens(doc)

    if len(tokens) != 2:
        return None, None, None

    t1 = tokens[0]
    t2 = tokens[1]

    #save the surface version (as it is)
    surface = t1.text.lower() + " " + t2.text.lower()

    #save the lemma version
    lemma = t1.lemma_.lower() + " " + t2.lemma_.lower()

    #save the pos of the phrase
    pos = (t1.pos_.upper(), t2.pos_.upper())

    return surface, lemma, pos


#this function checks if the final phrases overlap. If that's the case we remove the lower score one
def prune_overlaps(items):
    
    #we remove phrases that overlap with each other
    kept = []

    
    sorted_items = sorted(items, key=lambda x: x["score"], reverse=True) #sort phrases so the high score one comes first

    for item in sorted_items:
        candidate = item["phrase_surface"]
        keep_item = True

        #check if this phrase overlaps with something we already kept
        for other in kept:
            other_text = other["phrase_surface"]

            # if there is an overlap, we don't add it 
            if candidate in other_text or other_text in candidate:
                keep_item = False
                break

        if keep_item:
            kept.append(item)

    return kept

#main phrase extraction function:
def extract_phrases(text, max_candidates=C.KEYBERT_MAX_CANDIDATES, score_cutoff=C.KEYBERT_SCORE_CUTOFF):

    if not text or text.strip() == "":
        return []

    nlp = get_spacy()
    kb = get_keybert()
    vectorizer = get_vectorizer()

    #keyBert extracts the raw candidate phrases
    raw_candidates = kb.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        use_mmr=True,
        diversity=0.7,
        top_n=max_candidates,
        vectorizer=vectorizer,
    )

    best_phrases = {}

    for phrase_text, score in raw_candidates:
        # cut the noisy phrases
        if float(score) < score_cutoff:
            continue

        #we clean the phrase
        cleaned = tidy_phrase(phrase_text)
        if cleaned == "":
            continue

        # spaCy processes the cleaned phrase
        doc = nlp(cleaned)

        # turn the phrase into a bigram
        surface, lemma, pos = lemma_bigram(doc)
        if surface is None:
            continue

        #we keep the POS tags that we defined earlier
        if pos not in ALLOWED_BIGRAMS:
            continue

        #we keep the best-scoring phrase per lemma
        if lemma not in best_phrases:
            best_phrases[lemma] = {
                "phrase_surface": surface,
                "phrase_lemma": lemma,
                "score": float(score),
                "pos_signature": pos[0] + " " + pos[1],
            }
        else:
            #we only replace if this score is better
            if float(score) > best_phrases[lemma]["score"]:
                best_phrases[lemma] = {
                    "phrase_surface": surface,
                    "phrase_lemma": lemma,
                    "score": float(score),
                    "pos_signature": pos[0] + " " + pos[1],
                }

    #finally we remove overlapping phrases
    final_phrases = list(best_phrases.values())
    final_phrases = prune_overlaps(final_phrases)

    return final_phrases

### Embedding and Storing functions
def embed_texts(texts):
   
    if not texts:
        return []

    model = get_embedder()

    #we encode the texts into vectors
    vectors = model.encode(
        texts,
        batch_size=C.EMBED_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # and then convert each numpy vector into a python list (float32)
    output = []
    for v in vectors:
        v = v.astype(np.float32)
        output.append(v.tolist())

    return output
    
def df_to_sqlite(df, sqlite_path, table, if_exists="replace"):

    sqlite_path = Path(sqlite_path) #get the database

    #we create the folder if it's not there
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    #write the table
    with sqlite3.connect(str(sqlite_path)) as conn:
        df.to_sql(table, conn, if_exists=if_exists, index=False)

def as_json(value):
    #convert the python object into JSON string (keeping the utf-8 characters)
    return json.dumps(value, ensure_ascii=False)



def find_span_in_text(text, fragment, allow_fuzzy=False, min_tokens_found=1): 

    if not isinstance(text, str) or not isinstance(fragment, str):
        return (None, None)
        
    text_clean = text.strip()
    frag_clean = fragment.strip()

    if text_clean == "" or frag_clean == "":
        return (None, None)

    text_low = text_clean.lower()
    frag_low = frag_clean.lower()
    #exaxt find
    exact_pos = text_low.find(frag_low)

    if exact_pos != -1:
        start = exact_pos
        end = exact_pos + len(frag_clean)
        return (start, end)

    if not allow_fuzzy:
        return (None, None)

    #fuzzy find
    parts = frag_clean.split()
    tokens = []
    for w in parts:
        w2 = w.strip().lower()
        if w2 != "":
            tokens.append(w2)
            
    if len(tokens) == 0:
        return (None, None)

    positions = []

    for tok in tokens:
        pos = text_low.find(tok)
        if pos != -1:
            start_pos = pos
            end_pos = pos + len(tok)
            positions.append((start_pos, end_pos))

    if len(positions) < min_tokens_found:
        return (None, None)

    start_values = []
    end_values = []

    for (s, e) in positions:
        start_values.append(s)
        end_values.append(e)

    start = min(start_values)
    end = max(end_values)

    return (start, end)



