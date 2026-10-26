# Lab 15 — Answers (Problem-Solving Experience)

> Numbers marked **▶ fill from your Colab output** depend on the real 20 Newsgroups run. The notebook prints every one of them in Step 4.

---

## Challenge 1: Conversational Filler & Numeric Noise

**Why does `know` still get a substantial weight at 25% document frequency?**

TF-IDF is `TF(t,d) × log(N / DF_t)`. A word in 25% of posts has `N / DF = 4`, so `log(4) ≈ 1.39` (natural log). With scikit-learn's smoothed IDF, `ln((1+N)/(1+DF)) + 1 ≈ 2.39`. That is far from zero. IDF only goes to zero when a word is in *every* document, and only gets small when it is in nearly all of them. A word spread over a quarter of the corpus is still treated as moderately informative. And the weight is multiplied by TF, so a post that says "I know… you know… do you know" scores high on `know`. Standard English stopword lists don't contain `know`, `think`, `thanks`, `use`, so nothing removes them. They are common in forum posts, but not common enough for IDF to push them out.

**Refinement:**

```python
TfidfVectorizer(token_pattern=r'\b[a-zA-Z]{3,}\b', max_df=0.5, stop_words=custom_stopwords)
```

- `token_pattern` drops numbers and 1–2 letter fragments (`123`, `ve`, `ll`).
- `max_df=0.5` drops words appearing in more than half the posts. It does **not** catch `know` (25%), so the custom stopword list is still needed.
- `custom_stopwords` removes conversational filler explicitly.

**How the cluster top words change:** *▶ fill from your output* (compare "Top 8 defining words per cluster (baseline)" with "CLEANED"). Expected pattern: filler words and number-like tokens disappear from the centroids, replaced by topic terms (e.g. `engine`, `car`, `patient`, `drug`, `mac`, `circuit`), so the clusters read as clearer topics.

---

## Challenge 2: Hardware Vocabulary Overlap

**Which two categories overlap the most?** `sci.electronics` and `comp.sys.mac.hardware`. In `pd.crosstab(df['true_label'], df['cluster'])`, these two rows are the ones that spread across the same clusters, while `rec.autos` and `sci.med` each land mostly in their own cluster. *▶ confirm with your crosstab and the row-normalised table.*

**Shared technical words pulling them together:** the notebook prints the words that are heavily weighted in *both* categories. Typical ones are `board`, `chip`, `power`, `drive`, `monitor`, `card`, `memory`, `bus`. *▶ list at least 3 from your "Technical words shared…" output.* A Mac hardware post about a logic board and an electronics post about a circuit board both rely on this same hardware vocabulary, so single-word TF-IDF vectors look similar.

**How bigrams (`ngram_range=(1, 2)`) help:** unigrams are ambiguous: `board`, `power`, `drive` appear in both categories. Bigrams add context. `power supply` and `logic board` ground the word in a specific sense, and Mac-specific phrases (`power mac`, `hard drive`, `mac ii`) separate Mac posts from general electronics phrases (`power supply`, `op amp`, `signal noise`). The feature space then has terms that belong to one category far more than the other, which pulls the centroids apart. *▶ compare the crosstab and Adjusted Rand Index before and after bigrams in your output.*

---

## Challenge 3: Rant Length Bias (BoW vs TF-IDF)

The notebook prints a table of the L2 norm `‖x‖₂` of the longest and shortest post:

| Post | ‖x‖₂ in `X_bow` | ‖x‖₂ in `X_tfidf` |
|---|---|---|
| Longest | *▶ fill (large)* | 1.0 |
| Shortest | *▶ fill (small)* | 1.0 |

**Why KMeans fails on raw BoW:** BoW values are raw counts, so a vector's length grows with the length of the post. KMeans uses Euclidean distance. A 2,000-word rant sits very far from the origin, and a 20-word question sits very close to it, whatever their topics. Clusters end up grouping posts by **length** rather than **topic**, and long posts dominate the centroid averages.

**How TF-IDF fixes it:** `TfidfVectorizer` applies L2 normalisation by default (`norm='l2'`), so every document vector has length exactly 1. Only the *direction* (which words, in what proportion) remains. A short question and a long rant about the same topic point the same way, and Euclidean distance between unit vectors is monotonic in cosine similarity, so KMeans now clusters by topic.

---

## Challenge 4: Regulated Compliance Explainability

`explain_document(doc_id=412, top_n=5)` is implemented in the notebook. It returns a list of `(term, weight)` tuples for the document's top 5 non-zero TF-IDF features, and the audit cell prints them together with the document's true label and assigned cluster. *▶ paste the printed 5 terms and weights here.*

**Justification (3 sentences):**

TF-IDF produces a document vector whose every dimension is a named word with an exact numeric weight, so an auditor can see precisely which terms placed Document #412 in its cluster and reproduce that result identically on every run. A black-box LLM cannot give a faithful, deterministic account of why it routed a document, which is a problem when a regulator requires traceability and reproducibility for patient adverse-event reports. TF-IDF is also cheap, locally runnable and free of data-leakage risk to a third-party model, which makes it a defensible transparent baseline against which any more complex model must be justified.
