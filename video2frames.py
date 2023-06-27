import cv2
import os

def video_to_frames(video_path, save_path, interval=2):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            # 현재 프레임 인덱스가 프레임 간격으로 나누어 떨어지면, 이미지를 저장함
            if frame_idx % frame_interval == 0:
                save_img_path = f"{save_path}/frame_{frame_idx//frame_interval}.jpg"
                cv2.imwrite(save_img_path, frame)
            frame_idx += 1

            if frame_idx >= total_frames:
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

#경로는 변경하여 사용
if not os.path.exists("/root/video_labelling/frames/"):
    os.makedirs("/root/video_labelling/frames/")

for i in range(31,51):
    if not os.path.exists(f"/root/video_labelling/frames/video_{i}/"):
        os.makedirs(f"/root/video_labelling/frames/video_{i}/")
    video_to_frames(f"/root/video_labelling/videos/video_{i}.mp4", f"/root/video_labelling/frames/video_{i}/")
