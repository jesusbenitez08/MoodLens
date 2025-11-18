from flask import Flask, render_template, request
from main import get_news, analyze_sentiments, show_chart
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    mood_data = None
    articles = []
    chart_path = None
    topic = ""
    summary = None

    if request.method == "POST":
        topic = request.form.get("topic", "").strip()

        if topic:
            articles = get_news(topic)
            if articles:
                mood_counts, analyzed, summary = analyze_sentiments(articles)
                chart_path = show_chart(mood_counts, topic)

                mood_data = {
                    "results": mood_counts,
                    "analyzed": analyzed,
                }

    return render_template(
        "index.html",
        mood_data=mood_data,
        articles=articles,
        chart_path=chart_path,
        topic=topic,
        summary=summary,
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
