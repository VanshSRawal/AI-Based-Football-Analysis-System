# ⚽ AI-Powered Football Video Analysis System

A computer vision pipeline using **YOLOv8**, **object tracking**, and advanced analytics to extract insights from football match footage. This project detects players, referees, and the ball, and analyzes their movements, speeds, team assignments, and ball possession.

---

## 📁 Project Structure

```
.
├── main.py                            # End-to-end pipeline to process video
├── yolo_inference.py                 # YOLOv8 object detection demo
├── yolo_football_training.ipynb      # Google Colab training notebook
├── accuracy_report.csv               # Accuracy metrics in tabular form
├── evaluation_accuracy_summary.png   # Accuracy summary bar chart
├── player_visibility_chart.png       # Player-wise visibility chart
├── README.md                         # Project documentation
├── camera_movement_estimator/        # Camera motion estimation logic
├── development_and_analysis/         # Optional dev notebooks/scripts
├── evaluate_accuracy/                # Accuracy analysis scripts
├── input_videos/                     # [Excluded] Input football match clips
├── models/                           # [Excluded] Trained YOLOv8 model weights
├── output_videos/                    # [Excluded] Annotated output video
├── player_ball_assigner/             # Ball-player assignment logic
├── runs/                             # YOLO training outputs (if any)
├── screenshots/                      # Demo visuals or logs
├── speed_and_distance_estimator/     # Speed & distance calculation
├── stubs/                            # Cached track/motion data (pickles)
├── team_assigner/                    # Team clustering using jersey colors
├── trackers/                         # YOLO + ByteTrack tracking pipeline
├── training/                         # Training code or configs
├── utils/                            # Utility functions
├── view_transformer/                 # Perspective view transformation
```

---

## 🧠 Key Features

| Feature               | Description |
|-----------------------|-------------|
| 🎯 Player Tracking     | YOLOv8 + ByteTrack used for ID-level tracking |
| ⚽ Ball Possession     | Assigned based on proximity of players to ball |
| 🟦 Team Identification | KMeans color clustering from jerseys |
| 📉 Speed & Distance    | Calculated using frame rate and perspective mapping |
| 📊 Accuracy Evaluation | Measures consistency of tracking and labeling |
| 🎥 Visual Overlays     | Draws player ellipses, team color, possession arrows, etc. |

---

## 🔍 Dataset Used

- **Source:** [Roboflow Universe - Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)
- Contains labeled images of `player`, `referee`, and `ball`
- Used to train a custom YOLOv8 model for this project

---

## 🧪 Model Training

Model training was performed using Google Colab:  
➡️ [Open Training Notebook](https://colab.research.google.com/drive/1lTmvPfDC65MEhIXXx2zfsZkPe20_jD8G?usp=sharing)

- Framework: YOLOv8
- Classes: player, referee, ball
- Weights: `best.pt`, `last.pt` (excluded from repo due to size)

---

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Add necessary assets manually:**
   - `models/best.pt` (YOLOv8 weights)
   - `input_videos/Clip1.mp4` (your match video)

4. **Run the pipeline**
```bash
python main.py
```

5. **Evaluate accuracy**
```bash
python evaluate_accuracy/evaluate_accuracy.py
```

---

## 📈 Accuracy Metrics Output

- **Ball Possession Accuracy:** `100.0%`
- **Player Tracking Consistency:** `60.6%`
- **Team Assignment Accuracy:** `97.8%`

Two visuals are also generated:
- `evaluation_accuracy_summary.png`
- `player_visibility_chart.png`

---

## 📌 Notes

- Input/output videos and model weights are excluded due to GitHub's 25MB limit.
- The dataset is publicly available and the training notebook can reproduce the model.

---

## 📝 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Universe Dataset](https://universe.roboflow.com/)
- [Google Colab](https://colab.research.google.com/)
