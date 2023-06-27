import pickle
import pprint

dir = "/root/video_labelling/output/slowfast"   #경로는 변경하여 사용

for i in range(31, 51):
    obj = pickle.load(open(f"{dir}/video_{i}_output.pkl", "rb"))
    with open(f"{dir}/video_{i}_output.txt", "w") as f:
        pprint.pprint(obj, stream=f)