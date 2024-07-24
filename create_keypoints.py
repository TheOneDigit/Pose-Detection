import cv2
from ultralytics import YOLO
import pandas as pd
import os 


def create_keypoints(video_url: str):
  current_dir = os.getcwd()
  print('This is the directory: ', current_dir)
  print('This is video URL: ', video_url)

  model = YOLO(f"{current_dir}/yolov8n-pose.pt")
  video_path = video_url
  cap = cv2.VideoCapture(video_path)
  frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
  fps = cap.get(cv2.CAP_PROP_FPS)
  print('This is fps: ', fps)
  seconds = round(frames/fps)

  frame_total = 500 
  i = 0
  a = 0
  all_data = []
  while (cap.isOpened()):
    cap.set(cv2.CAP_PROP_POS_MSEC, (i * ((seconds/frame_total)*1000)))
    flag, frame = cap.read()

    if flag == False:
      break


    image_path = f'{current_dir}/images/img_{i}.jpg'
    os.makedirs(image_path, exist_ok=True)
    
    cv2.imwrite(image_path, frame)

    # YOLOv8 Will detect your video frame
    results = model(frame, verbose=False)

    for r in results:
      bound_box = r.boxes.xyxy  # get the bounding box on the frame
      conf = r.boxes.conf.tolist() # get the confident it is a human from a frame
      keypoints = r.keypoints.xyn.tolist() # get the every human keypoint from a frame

      # this code for save every human that detected from 1 image, so if 1 image have 10 people, we will save 10 human picture.

      for index, box in enumerate(bound_box):
        if conf[index] > 0.75: # we do it for reduce blurry human image.
          x1, y1, x2, y2 = box.tolist()
          pict = frame[int(y1):int(y2), int(x1):int(x2)]
          output_path = f'{current_dir}/images_human/person_{a}.jpg'
          os.makedirs(output_path, exist_ok=True)
          # we save the person image file name to csv for labelling the csv file.
          data = {'image_name': f'person_{a}.jpg'}

          # Initialize the x and y lists for each possible key
          for j in range(len(keypoints[index])):
              data[f'x{j}'] = keypoints[index][j][0]
              data[f'y{j}'] = keypoints[index][j][1]

        # we save the human keypoint that detected by yolo model to csv file to train our Machine learning model later.

          all_data.append(data)
          cv2.imwrite(output_path, pict)
          a += 1

    i += 1
  print(i-1, a-1)
  cap.release()
  cv2.destroyAllWindows()
  df = pd.DataFrame(all_data)
  csv_file_path = f'{current_dir}/keypoints.csv'
  df.to_csv(csv_file_path, index=False)

  return 'keypoints.csv created successfully'
