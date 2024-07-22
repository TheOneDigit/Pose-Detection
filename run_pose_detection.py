import cv2
from ultralytics import YOLO
import xgboost as xgb
import pandas as pd

def run_poseDetection(video_path: str):
  model_yolo = YOLO('/content/yolov8n-pose.pt')
  model = xgb.Booster()
  model.load_model('/content/model_weights.xgb')
  video_path = video_path
  cap = cv2.VideoCapture(video_path)
  # print('Total Frame', cap.get(cv2.CAP_PROP_FRAME_COUNT))

  fps = int(cap.get(cv2.CAP_PROP_FPS))
  width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Define the codec and create VideoWriter object
  # Actually is optional, if you dont need to save your detection you can remove the 3 line of below code.
  fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
  output_path = "output_level4_xgb.avi"
  out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


  frame_tot = 0
  # Loop through the video frames
  while cap.isOpened():
      # Read a frame from the video
      success, frame = cap.read()

      if success:
          # Run YOLOv8 inference on the frame
          results = model_yolo(frame, verbose = False)

          # Visualize the results on the frame
          annotated_frame = results[0].plot(boxes = False)

          for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            for index, box in enumerate(bound_box):
              if conf[index] > 0.75:
                  x1, y1, x2, y2 = box.tolist()
                  data = {}

                  # Initialize the x and y lists for each possible key
                  for j in range(len(keypoints[index])):
                      data[f'x{j}'] = keypoints[index][j][0]
                      data[f'y{j}'] = keypoints[index][j][1]

                  df = pd.DataFrame(data, index=[0])
                  dmatrix = xgb.DMatrix(df)
                  cut = model.predict(dmatrix)
                  binary_predictions = (cut > 0.5).astype(int)
                  if binary_predictions == 0:
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, 'Cutting', (int(x1), int(y1)), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,0,0), 3)

          # Display the annotated frame
          # cv2.imshow(annotated_frame)

          # save the video (remove it if you dont want to save video result)
          out.write(annotated_frame)
          frame_tot += 1
          # print('Processed Frame : ', frame_tot)

          # Break the loop if 'q' is pressed
          if cv2.waitKey(1) & 0xFF == ord("q"):
              break
      else:
          # Break the loop if the end of the video is reached
          break

  # Release the video capture object and close the display window

  cap.release()
  cv2.destroyAllWindows()

  return f'Pose Detection has done successfully with file name {output_path}'