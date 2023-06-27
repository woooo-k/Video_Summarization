import collections
import pickle
import pprint

#경로는 변경하여 사용
dir = "/root/video_labelling/output/videomaev2/"

for i in range(31,51):
    with open(dir+f"video_{i}_output.pkl", "rb") as f:
        data = pickle.load(f)

    class_counter = collections.Counter()

    for key in data:
        classes, _ = data[key]
        class_counter.update(classes)

    # 가장 많이 등장한 5개의 class 출력
    most_common_classes = class_counter.most_common(5)

    with open(f"/root/video_labelling/output/videomaev2/video_{i}_class.txt", "w") as f:
        pprint.pprint(most_common_classes, stream=f)
        print(most_common_classes)
