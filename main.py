import requests
from transformers import pipeline
import matplotlib.pyplot as plt

# set up

# NewsAPI key
NEWS_API_KEY = "da63db32ee894ccda151c8b9d75c39c4"

# hugging face sentiment analysis like the pipeline -test
sentiment_analyzer = pipeline("sentiment-analysis")

# articles
def get_news(topic, page_size=10):
    """Fetch recent articles from NewsAPI for the given topic."""
    url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&pageSize={page_size}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Error fetching news: {response.status_code}")
        print(response.text)
        return []

    data = response.json()
    return data.get("articles", [])

# analyze sentiments and emotions
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

        # narrow labels to just 3 moods for now
        if label == "POSITIVE":
            results["POSITIVE"] += 1
        elif label == "NEGATIVE":
            results["NEGATIVE"] += 1
        else:
            results["NEUTRAL"] += 1

        analyzed.append((title, label, round(result["score"], 2)))

    return results, analyzed

# show results
def show_chart(results, topic):
    """Display sentiment breakdown in a bar chart."""
    moods = list(results.keys())
    counts = list(results.values())

    plt.bar(moods, counts, color=["green", "red", "gray"])
    plt.title(f"Mood Breakdown for '{topic}'")
    plt.xlabel("Mood")
    plt.ylabel("Number of Articles")

    #save chart (test for now-willshow later)
    output_file = f"{topic}_mood_chart.png"
    plt.savefig(output_file)
    print(f"\nChart saved as {output_file}")



# main program first test

if __name__ == "__main__":
    print("=== MoodLens - Sentiment Research Tool ===")
    topic = input("Enter a topic to analyze: ").strip()

    if not topic:
        print("Please enter a valid topic.")
        exit()

    print(f"\nFetching news for '{topic}'...")
    articles = get_news(topic)

    if not articles:
        print("No articles found or API limit reached.")
        exit()

    print(f"Analyzing {len(articles)} articles...")
    mood_counts, analyzed_articles = analyze_sentiments(articles)

    #quick summary line 
    total = sum(mood_counts.values())
    if total > 0:
        positive_pct = round((mood_counts["POSITIVE"] / total) * 100, 1)
        print(f"\nThis topic's mood today is approximately {positive_pct}% positive.\n")

    #resutls
    print("--- Sentiment Breakdown ---")
    for mood, count in mood_counts.items():
        print(f"{mood}: {count}")

    print("\n--- Article Mood Summary ---")
    for title, mood, score in analyzed_articles:
        print(f"[{mood}] ({score}) {title}")

    # chart
    show_chart(mood_counts, topic)
