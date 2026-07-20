import json

INPUT_FILE = "data/raw/Electronics.jsonl"
OUTPUT_FILE = "data/raw/electronics_sample.jsonl"

N_REVIEWS = 50_000


with open(INPUT_FILE, "r") as infile, open(OUTPUT_FILE, "w") as outfile:
    for i, line in enumerate(infile):
        if i >= N_REVIEWS:
            break

        review = json.loads(line)
        outfile.write(json.dumps(review) + "\n")

print(f"Saved {N_REVIEWS:,} reviews to {OUTPUT_FILE}")