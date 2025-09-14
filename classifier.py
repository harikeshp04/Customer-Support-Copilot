# classifier.py
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# topics required by challenge:
TOPICS = ["How-to","Product","Connector","Lineage","API/SDK","SSO","Glossary","Best practices","Sensitive data"]

# instantiate heavy models once
_zsl = None
_sent_analyzer = None

def _get_zsl():
    global _zsl
    if _zsl is None:
        _zsl = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _zsl

def _get_sentiment_analyzer():
    global _sent_analyzer
    if _sent_analyzer is None:
        _sent_analyzer = SentimentIntensityAnalyzer()
    return _sent_analyzer

def classify_ticket(text, threshold=0.25):
    """
    Returns dict:
      {tags: [str], topic_scores: [{label,score}], sentiment: str, sentiment_compound: float, priority: str}
    """
    text = str(text)
    zsl = _get_zsl()
    sent = _get_sentiment_analyzer()

    # zero-shot; allow multiple labels
    res = zsl(text, candidate_labels=TOPICS, multi_label=True)
    labels = []
    for lab, sc in zip(res['labels'], res['scores']):
        if sc >= threshold:
            labels.append({"label": lab, "score": float(sc)})
    if not labels:
        # fallback: pick the top label
        labels = [{"label": res['labels'][0], "score": float(res['scores'][0])}]

    tags = [l['label'] for l in labels]

    # sentiment mapping using VADER compound score
    s = sent.polarity_scores(text)
    comp = s['compound']
    if comp <= -0.5:
        sentiment = "Angry"
    elif comp < -0.05:
        sentiment = "Frustrated"
    elif comp <= 0.05:
        sentiment = "Neutral"
    else:
        sentiment = "Curious"

    # simple priority rules (customize as needed)
    priority = "P2"
    lower = text.lower()
    if "sensitive" in " ".join(tags).lower() or "sensitive data" in lower or "data leak" in lower:
        priority = "P0"
    elif any(k in lower for k in ["outage", "down", "data loss", "failed", "500", "urgent", "blocker", "blocking"]):
        priority = "P0"
    elif sentiment in ["Angry", "Frustrated"]:
        priority = "P1"

    return {"tags": tags, "topic_scores": labels, "sentiment": sentiment, "sentiment_compound": comp, "priority": priority}
