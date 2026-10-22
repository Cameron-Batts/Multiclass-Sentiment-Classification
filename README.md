# Multi-Class Sentiment Classification

**Author:** Cameron Batts  
**Tools:** Python · pandas · scikit-learn · NLTK · TextBlob

## Validation update - October 22, 2026

| Model | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Naive Bayes | 0.5520 | 0.4778 | 0.5067 |
| Logistic regression | 0.5605 | 0.5442 | 0.5581 |
| TextBlob fixed | 0.4136 | 0.4006 | 0.3894 |
| TextBlob validation-selected | 0.4823 | 0.4787 | 0.4782 |

All 18 cells ran on 40,000 posts with 42,633 vocabulary features. Thresholds selected on training-derived validation data were -0.05 and 0.10. The tuned TextBlob weighted F1 was 0.4782; it improved on the fixed rule but remained below the trained classifiers. The original subjective labels, preprocessing and non-stratified split were retained. No patient-feedback outcome is claimed.

[Aggregate validation record](reports/validation_results.json). [Repeatable validation script](validation/rerun_analysis.py). Run `python validation/rerun_analysis.py` from the project root. Install `requirements-validation.txt` first; source CSVs stay local.

## Business Problem
Classify short text into negative, neutral, and positive sentiment while retaining visibility into class-level errors. This project compares trained classifiers against a fixed lexicon-based baseline on public social-media data.

## Project Approach
- Map 13 emotion categories across 40,000 posts into three sentiment classes.
- Clean text and split 80/20 with random_state=42.
- Fit a 42,633-feature bag-of-words vocabulary on training text only.
- Compare Multinomial Naive Bayes, logistic regression and TextBlob on the same test split.
- Inspect per-class precision, recall, and F1 alongside weighted metrics.

## Historical findings and original context
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
The cleaned workflow was re-executed; current aggregate metrics are recorded separately from the original reports. The threshold search was executed with separate positive/negative cutoffs selected on training-derived validation data; fresh test results are recorded. Tested numerical/NLP packages are recorded in requirements-validation.txt.

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
