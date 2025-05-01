import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Load track data
with open("stubs/track_stubs.pkl", "rb") as f:
    tracks = pickle.load(f)

FRAME_SAMPLE_RATE = 10
TRACKING_THRESHOLD = 0.4

def ball_possession_accuracy(tracks):
    correct = 0
    total = 0
    for frame_num in range(0, len(tracks['players']), FRAME_SAMPLE_RATE):
        player_track = tracks['players'][frame_num]
        for pid, pdata in player_track.items():
            if 'has_ball' in pdata:
                total += 1
                if pdata['has_ball']:
                    correct += 1
    return 100 * correct / total if total > 0 else 0

def player_tracking_consistency(tracks):
    frame_ids = [i for i in range(0, len(tracks['players']), FRAME_SAMPLE_RATE)]
    player_frames = defaultdict(list)

    for frame_num in frame_ids:
        for player_id in tracks['players'][frame_num]:
            player_frames[player_id].append(frame_num)

    consistent_players = []
    player_visibility = {}

    for pid, frames in player_frames.items():
        ratio = len(frames) / len(frame_ids)
        player_visibility[pid] = ratio * 100
        if ratio > TRACKING_THRESHOLD:
            consistent_players.append(pid)

    return (100 * len(consistent_players) / len(player_frames)
            if player_frames else 0), player_visibility

def team_assignment_accuracy(tracks):
    total = 0
    consistent = 0
    player_team = {}

    for frame in range(0, len(tracks['players']), FRAME_SAMPLE_RATE):
        for pid, pdata in tracks['players'][frame].items():
            team = pdata.get('team', None)
            if team in [1, 2]:
                if pid in player_team:
                    if player_team[pid] == team:
                        consistent += 1
                else:
                    player_team[pid] = team
                total += 1

    return 100 * consistent / total if total > 0 else 0

# Run Evaluations 
print("Evaluating Final Video Annotations...\n")

possession_acc = ball_possession_accuracy(tracks)
tracking_acc, player_visibility = player_tracking_consistency(tracks)
team_acc = team_assignment_accuracy(tracks)

# Print player-level breakdown
for pid, visibility in sorted(player_visibility.items(), key=lambda x: -x[1]):
    print(f"Player {pid}: visible in {visibility:.1f}% of frames")

# Print results
print(f"\nBall Possession Accuracy:     {possession_acc:.2f}%")
print(f"Player Tracking Consistency:  {tracking_acc:.2f}%")
print(f"Team Assignment Accuracy:     {team_acc:.2f}%")

# Save CSV
df = pd.DataFrame([{
    "Ball Possession Accuracy (%)": possession_acc,
    "Player Tracking Consistency (%)": tracking_acc,
    "Team Assignment Accuracy (%)": team_acc
}])
df.to_csv("accuracy_report.csv", index=False)
print("Report saved to 'accuracy_report.csv'")

# Save Plots 

# Overall Accuracy Chart
plt.figure(figsize=(6, 4))
plt.bar(["Ball Possession", "Tracking Consistency", "Team Assignment"],
        [possession_acc, tracking_acc, team_acc],
        color=["blue", "green", "purple"])
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.title("Evaluation Accuracy Summary")
plt.tight_layout()
plt.savefig("evaluation_accuracy_summary.png")
plt.close()

# Player Visibility Chart
plt.figure(figsize=(12, 6))
plt.bar([str(pid) for pid in player_visibility.keys()],
        list(player_visibility.values()), color="orange")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Visibility (%)")
plt.title("Per-Player Visibility Across Sampled Frames")
plt.tight_layout()
plt.savefig("player_visibility_chart.png")
plt.close()

print("Charts saved: 'evaluation_accuracy_summary.png', 'player_visibility_chart.png'")
