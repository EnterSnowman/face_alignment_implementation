from preprocessing_loading.loader import get_list_of_images_and_landmarks
import cv2 as cv
import numpy as np

from utils.bounding_box import get_bounding_boxes


def get_images_with_bounding_boxes(image_folder, image_format=".jpg", landmark_format=".pts", is_debug=False,
                                   debug_size=20):
    ims, lms = get_list_of_images_and_landmarks(image_folder, image_format, landmark_format, is_debug=is_debug,
                                                debug_size=debug_size)
    bounding_boxes = get_bounding_boxes(ims, lms)
    return ims, lms, bounding_boxes


def get_mean_shape_and_localized_landmarks(landmarks, bounding_boxes):
    localized_landmarks = []
    # localize
    for (image_landmark, bounding_box) in zip(landmarks, bounding_boxes):
        new_lm = image_landmark.copy()
        new_lm[:, 0] = (image_landmark[:, 0] - bounding_box[0]) / bounding_box[2] * 2 - 1
        new_lm[:, 1] = (image_landmark[:, 1] - bounding_box[1]) / bounding_box[3] * 2 - 1
        localized_landmarks.append(new_lm)

    localized_landmarks = np.array(localized_landmarks)

    # mean shape
    mean_shape = np.mean(localized_landmarks, axis=0)
    # print(mean_shape.shape)
    return mean_shape, localized_landmarks


def get_training_data(list_of_image_names, localized_landmarks, bounding_boxes, mean_shape,
                      max_image_size=300):
    rotations = []
    rotations_inv = []
    shifts = []
    shifts_inv = []
    pupil_distances = []
    images = []
    normalized_shapes = []
    estimated_shapes = []
    for image_name, shape, bounding_box in zip(list_of_image_names, localized_landmarks, bounding_boxes):
        # normalize shape
        mat = cv.estimateRigidTransform(shape, mean_shape, False)
        if mat is None:
            print(image_name, "mat is None")
            continue

        rotation = mat[:, :2]
        shift = mat[:, 2]
        normalized_shape = np.transpose(np.dot(rotation, shape.T) + shift[:, None], (1, 0))

        mat = cv.estimateRigidTransform(normalized_shape, shape, False)
        if mat is None:
            print(image_name, "mat inv is None")
            continue

        rotation_inv = mat[:, :2]
        shift_inv = mat[:, 2]

        # compute normalized inter-pupil distance
        right_eye_center = np.sum(shape[37:39] + shape[40:42], axis=0) / 4
        left_eye_center = np.sum(shape[43:45] + shape[46:48], axis=0) / 4
        pupil_distance = np.sqrt(np.sum((right_eye_center - left_eye_center) ** 2))

        rotations.append(rotation)
        rotations_inv.append(rotation_inv)
        shifts.append(shift)
        shifts_inv.append(shift_inv)
        pupil_distances.append(pupil_distance)

        image_rgb = cv.imread(image_name)
        image_gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
        image_gray = image_gray[bounding_box[1]:bounding_box[1] + bounding_box[3],
                     bounding_box[0]:bounding_box[0] + bounding_box[2]]
        if bounding_box[2] > max_image_size:
            image_gray = cv.resize(image_gray, (max_image_size, max_image_size))
        images.append(image_gray)
        normalized_shapes.append(normalized_shape)
        estimated_shapes.append(mean_shape.copy())

    return images, localized_landmarks, np.array(
        normalized_shapes), rotations, rotations_inv, shifts, shifts_inv, pupil_distances, np.array(estimated_shapes)


def get_training_data_without_normalization(list_of_image_names, localized_landmarks, bounding_boxes, mean_shape,
                                            max_image_size=300):
    images = []
    estimated_shapes = []
    for image_name, shape, bounding_box in zip(list_of_image_names, localized_landmarks, bounding_boxes):
        image_rgb = cv.imread(image_name)
        image_gray = cv.cvtColor(image_rgb, cv.COLOR_BGR2GRAY)
        image_gray = image_gray[bounding_box[1]:bounding_box[1] + bounding_box[3],
                     bounding_box[0]:bounding_box[0] + bounding_box[2]]
        if bounding_box[2] > max_image_size:
            image_gray = cv.resize(image_gray, (max_image_size, max_image_size))
        images.append(image_gray)
        estimated_shapes.append(mean_shape.copy())
    return images, np.array(localized_landmarks), np.array(estimated_shapes)


if __name__ == "__main__":
    ims, lms, bounding_boxes = get_images_with_bounding_boxes("..\data\helen\\trainset")
    mean_shape, localized_landmarks = get_mean_shape_and_localized_landmarks(lms, bounding_boxes)
    print(mean_shape)
