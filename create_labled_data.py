import os
import pandas as pd

def get_label(image_name, cutting_path, non_cutting_path):
    if image_name in os.listdir(cutting_path):
        return 'cutting'
    elif image_name in os.listdir(non_cutting_path):
        return 'non_cutting'
    else:
        return None  # If we cant find the image on the folders from step 4

def create_labled_data(df: pd.DataFrame, labled_images_folder: str):
  '''
  labled_images_folder: A folder where images are classify as person cutting or non_cutting something

  '''
  dataset_path = labled_images_folder
  cutting_path = os.path.join(dataset_path, 'cutting')
  non_cutting_path = os.path.join(dataset_path, 'non-cutting')

  df['label'] = df['image_name'].apply(lambda x: get_label(x, cutting_path, non_cutting_path))
  df.to_csv(f'dataset.csv', index=False)

  return "Labled CSV created successfully"