import requests
from transformers import pipeline
import matplotlib.pyplot as plt
import os

# NewsAPI key
NEWS_API_KEY = "da63db32ee894ccda151c8b9d75c39c4"

# sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# get news
def get_news(topic, page_size=10):
    """Fetch recent articles from NewsAPI for the given topic."""
    url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&pageSize={page_size}&language=en&apiKey={NEWS_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code}")
        print(response.text)
        return []

    data = response.json()
    return data.get("articles", [])

# analyze
def analyze_sentiments(articles):
    """Run sentiment analysis on article titles."""
    results = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    analyzed = []

    for article in articles:
        title = article.get("title", "")
        if not title:
            continue

        result = sentiment_analyzer(title)[0]
        label = result["label"]

        if label == "POSITIVE":
            results["POSITIVE"] += 1
        elif label == "NEGATIVE":
            results["NEGATIVE"] += 1
        else:
            results["NEUTRAL"] += 1

            # test for summary line
        analyzed.append({
            "title": title,
            "label": label,
            "score": round(result["score"], 2),
            "description": article.get("description", ""),
            "url": article.get("url", "#")
        })

    total=sum(results.values())
    summary=""
    if total>0:
        positive_pct=round((results["POSITIVE"]/total)*100,1)
        summary=f"News on this topic today is approximately {positive_pct}% positive."

    return results, analyzed, summary


# chart
def show_chart(results, topic):
    """Display sentiment breakdown in a bar chart."""
    moods = list(results.keys())
    counts = list(results.values())

    plt.figure(figsize=(6, 4))
    plt.bar(moods, counts, color=["#22c55e",  "#ef4444", "#fbbf24"])
    plt.title(f"Mood Breakdown for '{topic}'")
    plt.xlabel("Mood")
    plt.ylabel("Number of Articles")

    # save inside static/
    static_dir = os.path.join(os.getcwd(), "static")
    os.makedirs(static_dir, exist_ok=True)

    filename = f"{topic.lower()}_mood_chart.png"
    filepath = os.path.join(static_dir, filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()

    print(f"Chart saved as static/{filename}")
    return f"/static/{filename}"
