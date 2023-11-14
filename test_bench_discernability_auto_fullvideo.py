import cv2
import cv2
import numpy as np
import joblib
import pickle
from keras.models import Sequential
from keras.models import load_model
import torch
import urllib.request
import matplotlib.pyplot as plt
import os
import pandas as pd
from utility import calculate_centroid, Solution
import sys
#sys.setrecursionlimit(1000)
#Uncomment when using Windows if CuDNN does not work
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

device = torch.device("cuda") #if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def image_to_depth(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    return output





frameIntervals=[[0,100]]
frameIntervals.reverse()
resolutions=[100,150,200,250,300]
areaOfInterest_thresholds=[0,10,20,30,40,50,60,70,80,90]

for frameInterval in frameIntervals:
    discernability_results = np.zeros(shape=[len(resolutions), len(areaOfInterest_thresholds)])

    num_of_frames_limit = frameInterval[1]
    for resolutions_index in range(len(resolutions)):
        for areaOfInterest_thresholds_index in range(len(areaOfInterest_thresholds)):

            num_of_frames = 0
            predicted_count = 0
            unable_to_predict_count = 0
            cap = cv2.VideoCapture('CS_graphtheory_centre.mp4', apiPreference=cv2.CAP_MSMF)

            while num_of_frames < num_of_frames_limit:
                ret, frame = cap.read()

                if ret == False:
                    break
                if ret == True:
                    if num_of_frames % 1 != 0 or num_of_frames < frameInterval[0]:
                        num_of_frames = num_of_frames + 1
                        continue
                    num_of_frames = num_of_frames + 1
                    #print(num_of_frames)
                    frame = cv2.resize(frame, (resolutions[resolutions_index], resolutions[resolutions_index]))
                    depth_image = image_to_depth(frame)

                    solution1 = Solution(depth_image, areaOfInterest_thresholds[areaOfInterest_thresholds_index])
                    spaces_identified = solution1.identify_spaces_iterative()

                    if len(spaces_identified) == 0:
                        unable_to_predict_count = unable_to_predict_count + 1
                        centre_of_frame = [resolutions[resolutions_index], resolutions[resolutions_index]]
                        cv2.rectangle(frame, (50 + 10, 50 + 10), (50 + 15, 50 + 15), (0, 255, 0), 1)
                    else:
                        predicted_count = predicted_count + 1

                    for i in range(len(spaces_identified)):
                        centroid = calculate_centroid(spaces_identified[i])
                        centroid = np.rint(centroid).astype(int)

                        cv2.rectangle(frame, (centroid[1] - 5, centroid[0] - 5), (centroid[1] + 5, centroid[0] + 5),(0, 0, 255), 1)

                    dist = frame
                    dist = dist.astype(np.uint8)
                    frame = dist


                    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('video', 600, 600)
                    cv2.imshow("video", frame)


                    # videoFrames = np.concatenate((videoFrames, [frame]), axis=0)
                    # print(videoFrames.shape)

                    if (cv2.waitKey(10) & 0xFF == ord('q')):
                        break

            cap.release()
            cv2.destroyAllWindows()
            predicted_ratio = predicted_count / (predicted_count + unable_to_predict_count)
            discernability_results[resolutions_index][areaOfInterest_thresholds_index] = predicted_ratio
            print(discernability_results)

    print(discernability_results)
    ## convert your array into a dataframe
    df = pd.DataFrame(discernability_results)

    ## save to xlsx file

    filepath = 'CS_graphtheory_centre_Reso 100-300_Threshold 0-90_'
    dupe = filepath + str(frameInterval[0]) + " to " + str(frameInterval[1]) + ".xlsx"
    df.to_excel(dupe, index=False)
