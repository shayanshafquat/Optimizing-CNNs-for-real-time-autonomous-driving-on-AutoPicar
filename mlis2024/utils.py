import os
import cv2
import pandas as pd

def get_merged_df(data_dir, norm_csv_path):
    # Read the normalized CSV data
    df = pd.read_csv(norm_csv_path)

    # Initialize lists to store the data
    image_id = []
    image_path = []
    image_array = []
    file_size = []

    # List files in the specified directory
    file_list = os.listdir(data_dir)

    # Process each file in the directory
    for filename in file_list:
        # Read the image
        im = cv2.imread(os.path.join(data_dir, filename))

        # Append data to the lists
        image_id.append(int(filename.split('.')[0]))
        image_array.append(im)
        image_path.append(os.path.join(data_dir, filename))
        file_size.append(os.path.getsize(os.path.join(data_dir, filename)))

    # Create a DataFrame from the collected data
    data = {
        'image_id': image_id,
        'image': image_array,
        'image_path': image_path,
        'file_size': file_size
    }
    df_image = pd.DataFrame(data)

    # Merge the DataFrame with the CSV data
    merged_df = pd.merge(df, df_image, how='left', on='image_id')

    # Clean the merged DataFrame
    cleaned_df = merged_df[merged_df['speed'] <= 1]

    # Return the cleaned and merged DataFrame
    return cleaned_df
