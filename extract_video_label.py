import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
import json
import math
import os
import csv
from tqdm import tqdm
import pickle
from VideoMAEv2.models.modeling_finetune import vit_base_patch16_224
import time
# from VideoMAE.modeling_finetune import vit_base_patch16_224
##videomae, videomaev2는 github에서 다운받아서 사용
# videomae : https://github.com/MCG-NJU/VideoMAE
# videomaev2 : https://github.com/OpenGVLab/VideoMAEv2
# trained model은 model zoo에 있다.

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=False),
    ##model마다 normalize방법이 다르기 때문에 확인할 것!
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #videomae  v1, v2
    # transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]) #slowfast
    # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) #clip
])

def video2numpy(path, size = 224, sr=1):
    #video를 읽어와서 numpy로 변환

    vr = VideoReader(path, width=size, height=size) #HWC
    fps = int(round((vr.get_avg_fps())))
    frame_id_list = range(0, len(vr), sr)
    frames = vr.get_batch(frame_id_list).asnumpy()

    return frames, fps

def get_model(model_name = 'videomae') :
    #원하는 model 있으면 추가해서 사용
    num_frame = 16
    kinetics = 400

    if model_name == 'slowfast' :
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        num_frame = 32
        # model.blocks[6] = nn.Identity()

    elif model_name == 'videomaev2' :
        model = vit_base_patch16_224(num_classes = 710)
        checkpoint = torch.load('VideoMAEv2/vit_b_k710_dl_from_giant.pth', map_location="cuda")
        kinetics = 710
        model.load_state_dict(checkpoint["module"])
        # model.head = nn.Identity()


    elif model_name == 'videomae' :
        model = vit_base_patch16_224(num_classes = 400)
        checkpoint = torch.load('VideoMAE/checkpoint.pth', map_location="cuda")
        kinetics = 400
        model.load_state_dict(checkpoint["module"])
        # model.head = nn.Identity()
    else :
        exit()

    model.eval()
    model.cuda()

    return model, num_frame, kinetics

def extract_video(frames, fps, num_frames, save_path, model, model_name, kinetics_class) :
    sec = 2

    sampling_term = sec * fps
    seq_len = len(frames)
    sampling_list = [i for i in range(0, seq_len, sampling_term)]
    reminder = sampling_term - seq_len % sampling_term

    video = []
    for frame in frames :
        video.append(transform(frame))
    for _ in range(reminder) :
        video.append(torch.zeros(3, 224, 224))

    video = torch.stack(video, dim = 1).unsqueeze(0)
    feat = []
    idx = torch.linspace(0, sampling_term - 1, num_frames).long()
    with torch.no_grad() :

        for i in sampling_list :
            s = idx + i
            inputs = torch.index_select(video, 2, s)
            if model_name == 'slowfast' :
                fast = inputs.cuda()
                slow = fast[:, :, ::4].cuda()
                output = model([slow, fast]).cuda().squeeze()
                feat.append(output)

            elif model_name == 'videomaev2' :
                inputs = inputs.cuda()
                output = model(inputs).cuda().squeeze()
                feat.append(output)

            elif model_name == 'videomae' :
                inputs = inputs.cuda()
                output = model(inputs).cuda().squeeze()
                feat.append(output)

        output = torch.stack(feat, dim = 0)
        preds = F.softmax(output, dim = -1)
        value, indices = preds.topk(k=5)

        d = {}
        for k, index in enumerate(indices) :
            pred_class_names = [kinetics_class[int(i)] for i in index]
            d[k] = [pred_class_names, value[k]]
        
        # key : n번째 shot, value : class - score
        with open(save_path + '.pkl', 'wb') as f:
            pickle.dump(d, f)

parser = argparse.ArgumentParser()
parser.add_argument("--feature_extractor", type=str, default='videomae')

args = vars(parser.parse_args())
feat_name = args['feature_extractor']

dir = 'videos/'
model, num_frame, kinetics = get_model(feat_name)

kinetics_class = []
label_name = F'label_map_k{kinetics}.txt' ## class에 맞게 txt 변경
with open(label_name, 'r') as f :
    kinetics_class = f.read().splitlines()

print(f'[model : {feat_name}]')

start_time = time.time()

for i in tqdm(range(31, 51)) :
    video_key = f'video_{i}'
    tqdm.write(f'extract {video_key}.mp4')
    video_path = dir + video_key +'.mp4'
    save_path = "output/" + feat_name + '/' + video_key + '_output'
    directory = os.path.dirname(save_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    frames, fps = video2numpy(video_path)
    feat = extract_video(frames, fps, num_frame, save_path, model, feat_name, kinetics_class)

end_time = time.time()
execution_time = end_time - start_time

# 결과를 저장할 텍스트 파일 경로
time_file = "output/" + feat_name + '/' + "execution_time.txt"

# 실행 시간을 텍스트 파일에 저장
with open(time_file, "a") as f:
    f.write(f'[model : {feat_name}]\n')
    f.write(f"Execution time: {execution_time} seconds\n")
print(f"Execution time: {execution_time} seconds")