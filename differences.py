import pickle
import torch
import pprint
import os


def compare_pkls_activities(pkl1, pkl2):
    # Load the pkls
    with open(pkl1, 'rb') as f:
        data1 = pickle.load(f)
    with open(pkl2, 'rb') as f:
        data2 = pickle.load(f)

    differences = []

    # Compare each element
    for key in data1.keys():
        data1_values = data1[key]
        data2_values = data2.get(key)

        if data2_values is None:
            differences.append(f"Key {key} found in first pkl, but not in second")
        else:
            # Compare action labels
            data1_activities = set(data1_values[0])
            data2_activities = set(data2_values[0])

            activities_in_first_not_second = data1_activities - data2_activities
            activities_in_second_not_first = data2_activities - data1_activities

            if activities_in_first_not_second:
                differences.append([f"Activities in first pkl but not in second for key {key}: {activities_in_first_not_second}", len(activities_in_first_not_second)])
            if activities_in_second_not_first:
                differences.append([f"Activities in second pkl but not in first for key {key}: {activities_in_second_not_first}", len(activities_in_first_not_second)])

    for key in data2.keys():
        if key not in data1:
            differences.append(f"Key {key} found in second pkl, but not in first")

    return differences

# Usage
pkl1_path = '/root/video_labelling/output/slowfast/'
pkl2_path = '/root/video_labelling/output/videomae/'

if not os.path.exists("/root/video_labelling/output/differences/"):
    os.makedirs("/root/video_labelling/output/differences/")

res = []

for i in range(31, 51):
    print("================================================================================")
    differences = compare_pkls_activities(pkl1_path+f'video_{i}_output.pkl', pkl2_path+f'video_{i}_output.pkl')
    mean = 0
    cnt = 0
    if os.path.isfile(f"/root/video_labelling/output/differences/video_{i}_diff.txt"):
        os.remove(f"/root/video_labelling/output/differences/video_{i}_diff.txt")

    if differences:
        print(f"Differences were found in video_{i}_output:")
        for diff in differences:
            print(diff[0])
            print(f"cnt : {diff[1]}")
            with open(f"/root/video_labelling/output/differences/video_{i}_diff.txt", "a") as f:
                pprint.pprint(diff[0], stream=f)
                pprint.pprint(f"cnt : {diff[1]}", stream=f)
                mean += diff[1]
                cnt += 1
    else:
        print("No differences found.")

    with open(f"/root/video_labelling/output/differences/video_{i}_diff.txt", "a") as f:
        pprint.pprint(f"Total : {mean}", stream=f)
        pprint.pprint(f"Mean : {mean/cnt}", stream=f)
        print(f"Total : {mean}")
        print(f"Mean : {mean/cnt}")
        res.append([mean/cnt, i])
    print()

if os.path.isfile(f"/root/video_labelling/output/differences/res.txt"):
    os.remove(f"/root/video_labelling/output/differences/res.txt")
res.sort(reverse=True)
for r in res:
    with open(f"/root/video_labelling/output/differences/res.txt", "a") as f:
        pprint.pprint(f"video{r[1]} : {r[0]}", stream=f)
        print(f"video{r[1]} : {r[0]}")

