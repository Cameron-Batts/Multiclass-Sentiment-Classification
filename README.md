# Multi-Class Sentiment Classification

**Author:** Cameron Batts  
**Tools:** Python · pandas · scikit-learn · NLTK · TextBlob

## Business Problem
Classify short text into negative, neutral, and positive sentiment while retaining visibility into class-level errors. This project compares trained classifiers against a fixed lexicon-based baseline on public social-media data.

## Project Approach
- Map 13 emotion categories across 40,000 posts into three sentiment classes.
- Clean text and split 80/20 with random_state=42.
- Fit a 42,633-feature bag-of-words vocabulary on training text only.
- Compare Multinomial Naive Bayes, logistic regression and TextBlob on the same test split.
- Inspect per-class precision, recall, and F1 alongside weighted metrics.

## Key Results
| Model | Weighted precision | Weighted recall | Weighted F1 |
|---|---:|---:|---:|
| Naive Bayes | 0.5324 | 0.5520 | 0.5067 |
| Logistic regression | 0.5567 | 0.5604 | 0.5579 |
| TextBlob (±0.33) | 0.5780 | 0.4136 | 0.3894 |

Logistic regression improved weighted F1 by 0.0512 over Naive Bayes and improved neutral-class F1 from 0.1871 to 0.4057. Naive Bayes retained slightly higher negative-class F1. These are retained outputs from the original run, not newly computed results.

## Reproduce
```bash
python -m pip install -r requirements.txt
jupyter lab notebooks/multiclass_sentiment_classification.ipynb
```
Place a permitted copy of `text_emotion.csv` in `data/`. Required columns: `sentiment`, `content`. NLTK downloads English stopwords at runtime. The raw dataset is not distributed here; author identifiers and saved text excerpts have been omitted. `data/emotion_counts.csv` contains aggregate label counts only.

## Validation and Limitations
Code syntax checked; the cleaned notebook has not been re-executed. Baseline saved outputs were retained. An optional threshold search was revised to select independent positive/negative cutoffs on validation data; no new tuned results are claimed. Dependencies are unpinned.

The three-class label mapping is subjective. Negation/emoticons may be lost in preprocessing, username text may remain, and duplicate posts or authors can cross the random split. This is a social-media experiment, not measured patient-feedback performance.

## Data Source
CrowdFlower: https://data.world/crowdflower/sentiment-analysis-in-text . Retain source attribution and observe applicable data terms.

## Files
- `notebooks/multiclass_sentiment_classification.ipynb`: cleaned analysis and original numerical outputs.
- `reports/multiclass_sentiment_classification.html`: readable report.
- `reports/model_comparison.csv`: baseline test metrics.
- `data/emotion_counts.csv`: aggregate original emotion counts.
- `requirements.txt`: dependencies.

GitHub: https://github.com/Cameron-Batts  
Portfolio: https://cameron-batts.github.io
