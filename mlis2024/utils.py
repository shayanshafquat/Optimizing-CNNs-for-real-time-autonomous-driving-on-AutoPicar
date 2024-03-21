import os
import cv2
import pandas as pd
import shutil


def restructuring_data():
    # Load the CSV file with image IDs, angles, and speeds
    data = pd.read_csv('training_norm.csv')

    # Preprocess the data as per your provided logic
    data['angle'] = data['angle']*80 + 50
    data.loc[data['speed'] > 1, 'speed'] = 1

    # Base directory where images are stored
    image_dir = 'training_data/training_data/'

    # Create directories for angle and speed classes
    angle_class_dir = 'angle_class_data'
    speed_class_dir = 'speed_class_data'

    # Get unique classes for angles and speeds
    unique_angles = data['angle'].unique()
    unique_speeds = data['speed'].unique()

    # Make directories for angle classes
    for angle in unique_angles:
        angle_dir = os.path.join(angle_class_dir, str(angle))
        os.makedirs(angle_dir, exist_ok=True)

    # Make directories for speed classes
    for speed in unique_speeds:
        speed_dir = os.path.join(speed_class_dir, str(speed))
        os.makedirs(speed_dir, exist_ok=True)

    # Function to copy images to the respective class directories
    def copy_images_to_class_dirs(row, image_dir, class_dir, class_label):
        source_path = os.path.join(image_dir, f"{int(row['image_id'])}.png")
        target_path = os.path.join(class_dir, str(row[class_label]), f"{int(row['image_id'])}.png")
        if not os.path.exists(target_path):  # Check if the file has already been copied
            shutil.copy(source_path, target_path)

    # Copy images to angle class directories
    data.apply(lambda row: copy_images_to_class_dirs(row, image_dir, angle_class_dir, "angle"), axis=1)

    # Copy images to speed class directories
    data.apply(lambda row: copy_images_to_class_dirs(row, image_dir, speed_class_dir, "speed"), axis=1)

    print("Data restructuring complete.")


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

restructuring_data()
