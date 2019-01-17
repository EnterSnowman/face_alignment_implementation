import glob
import os
import numpy as np


def get_list_of_images_and_landmarks(image_folder, image_format=".jpg", landmark_format=".pts", is_debug=False,
                                     debug_size=20):
    landmarks = []
    list_of_images = []
    debug_counter = 0
    for landmark_path in glob.glob(os.path.join(image_folder, "*" + landmark_format)):
        if is_debug and debug_counter == debug_size:
            break
        # print(landmark_path)
        if os.path.exists(os.path.join(image_folder, os.path.basename(landmark_path).split(".")[0] + image_format)):
            list_of_images.append(
                os.path.join(image_folder, os.path.basename(landmark_path).split(".")[0] + image_format))
            landmarks.append(get_landmarks_as_numpy_array(landmark_path))
            debug_counter += 1
    return list_of_images, np.array(landmarks)


def get_landmarks_as_numpy_array(filename):
    landmarks = []
    # print(filename)
    with open(filename, "r") as lm:
        # print(lm.readlines())
        lines = lm.readlines()[3:]
        lines = lines[:-1]
        landmarks.extend([[float(coor) for coor in line[:-1].split(" ")] for line in lines])
        # print(landmarks)
    return np.array(landmarks)


if __name__ == "__main__":
    ims, lms = get_list_of_images_and_landmarks("..\data\helen\\trainset", is_debug=True)
