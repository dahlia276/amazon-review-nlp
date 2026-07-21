from src.openai_summarizer import summarize_reviews

reviews = [
    "This charger works great and charges my phone very quickly.",
    "Excellent build quality and good value for the price.",
    "The cable feels durable and has not frayed after months of use.",
    "Charging is fast, but the cable is a little stiff.",
    "Overall I'm very happy and would definitely buy it again.",
]

summary = summarize_reviews(reviews)

print("\nThemes:")
for theme in summary.themes:
    print(f"- {theme}")

print("\nPositives:")
for item in summary.positives:
    print(f"- {item}")

print("\nNegatives:")
for item in summary.negatives:
    print(f"- {item}")

print(f"\nOverall Sentiment: {summary.overall_sentiment}")

print("\nSummary:")
print(summary.summary)