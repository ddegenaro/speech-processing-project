import os
import json
import csv
from collections import defaultdict

# Load rankings
rankings = json.load(
    open(os.path.join('tts_eval', 'rankings.json'))
)

# CLAUDE'S IDEA
def calculate_elo_aggregate(all_rankings, initial_rating=1500, k=32):
    ratings = defaultdict(lambda: initial_rating)
    
    for prompt in all_rankings:
        for winner, loser in all_rankings[prompt]:
            w_rating = ratings[winner]
            l_rating = ratings[loser]
            
            expected_w = 1 / (1 + 10 ** ((l_rating - w_rating) / 400))
            ratings[winner] = w_rating + k * (1 - expected_w)
            ratings[loser] = l_rating + k * (0 - (1 - expected_w))
    
    return ratings

# Calculate ELO scores
final_scores = calculate_elo_aggregate(rankings)
final_scores_sorted = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

# Write to CSV
with open(os.path.join('tts_eval', 'elo_rankings.csv'), 'w+', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['model_name', 'elo'])
    for model, elo in final_scores_sorted:
        writer.writerow([model, elo])
