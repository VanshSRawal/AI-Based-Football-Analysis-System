import numpy as np
import cv2
from utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import pickle
import os

def main():
    # Step 1: Read video
    video_frames = read_video('input_videos/Clip1.mp4')

    # Step 2: Get object tracks
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    tracker.add_position_to_tracks(tracks)

    # Step 3: Camera Movement Estimation
    camera_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/camera_movement_stub.pkl'
    )
    camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Step 4: View Transform
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Step 5: Ball interpolation
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Step 6: Speed and Distance
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    # Step 7: Team Assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    print("\nAssigning teams to players...")
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    print("Team assignment complete.")

    # Step 8: Ball Possession
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    print("\nAssigning ball possession...")
    for frame_num, player_track in enumerate(tracks['players']):
        ball_data = tracks['ball'][frame_num].get(1)
        if not ball_data:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
            continue

        ball_bbox = ball_data['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            assigned_team = tracks['players'][frame_num][assigned_player].get('team')
            team_ball_control.append(assigned_team if assigned_team else 0)
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)

    print("Ball possession assignment complete.")
    team_ball_control = np.array(team_ball_control)

    # Optional: Save updated tracks
    with open('stubs/track_stubs.pkl', 'wb') as f:
        pickle.dump(tracks, f)
    print("Updated tracks saved to stubs/track_stubs.pkl")

    # Step 9: Draw final output
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_frames = camera_estimator.draw_camera_movement(output_frames, camera_movement_per_frame)
    output_frames = speed_estimator.draw_speed_and_distance(output_frames, tracks)

    save_video(output_frames, 'output_videos/output_video.avi')
    print("Final annotated video saved to output_videos/output_video.avi")

if __name__ == '__main__':
    main()
