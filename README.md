# Video_Summarization

frames/ : video2frames.py를 통해 생성된 디렉토리이다. 해당 디렉토리 안에는 각 input video별 2초 단위로 캡쳐된 이미지 파일들이 저장되어있다.
output/ : 모델의 실험 결과가 각 모델 이름의 디렉토리에 저장된다. 여기서 differences 디렉토리는 differences.py의 결과가 저장된다.
result/ : 모델이 평가한 output에 대한 나의 평가가 포함되어 있다.
differences.py: kinetic-400 dataset으로 pretrained 된 두 모델 SlowFast와 VideoMAE의 비교를 편히 하기 위해 사용된다.
most_class.py: 각 input video 마다 가장 많이 등장한 class top-5를 추려준다.
pkl2txt.py: .pkl 파일을 .txt 파일로 변환해준다.
video2frames.py: 수작업으로 output data를 평가할 때의 편의성을 위해 사용된다. input video별 2초 단위로 캡쳐된 이미지 파일들을 frames 디렉토리에 저장한다.

* extract_video_label.py: 메인으로 실험을 돌리는 코드이다.
