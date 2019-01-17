import numpy as np
import cv2 as cv


def get_sample_feature_locations(radii, num_features):
    all_feature_locations = []
    for r in radii:
        stage_feature_locations = []
        for i in range(num_features):
            single_feature_location = []
            # first point
            sample_r = r * np.random.uniform(0, 1)
            sample_theta = 2. * np.pi * np.random.uniform(0, 1)
            single_feature_location.extend([sample_r * np.cos(sample_theta), sample_r * np.sin(sample_theta)])

            # second point
            sample_r = r * np.random.uniform(0, 1)
            sample_theta = 2. * np.pi * np.random.uniform(0, 1)
            single_feature_location.extend([sample_r * np.cos(sample_theta), sample_r * np.sin(sample_theta)])

            stage_feature_locations.append(single_feature_location)
        all_feature_locations.append(stage_feature_locations)
    return np.array(all_feature_locations)


def get_dict_node_to_leaf_number(tree):
    node_to_leaf_dict = {}
    node_index = 0
    leaf_counter = 0
    for n in tree.tree_.__getstate__()['nodes']:
        if n[0] == -1 and n[1] == -1:
            node_to_leaf_dict[node_index] = leaf_counter
            leaf_counter += 1
        node_index += 1
    return node_to_leaf_dict, leaf_counter


def project_shape(shape, rotation_inv, shift_inv):
    shape_T = shape.T
    new_shape = np.dot(rotation_inv, shape_T)
    new_shape = new_shape + np.reshape(shift_inv, (2, 1))
    return new_shape.T


def compute_pixel_differences_for_single_image(projected_shape, image, sample_feature_locations, image_width,
                                               image_height, num_landmarks, stage):
    # print(projected_shape.shape)
    for i in range(num_landmarks):
        landmarks_differences = []

        landmark_x, landmark_y = projected_shape[i, 0], projected_shape[i, 1]
        for sample_feature_location in sample_feature_locations[stage, :, :]:
            # a
            loc_x_a = landmark_x + sample_feature_location[0]
            loc_y_a = landmark_y + sample_feature_location[1]

            pixel_x_a = (image_width / 2.0) + loc_x_a * (image_width / 2.0)
            pixel_y_a = (image_height / 2.0) + loc_y_a * (image_height / 2.0)
            # b
            loc_x_b = landmark_x + sample_feature_location[2]
            loc_y_b = landmark_y + sample_feature_location[3]

            pixel_x_b = (image_width / 2.0) + loc_x_b * (image_width / 2.0)
            pixel_y_b = (image_height / 2.0) + loc_y_b * (image_height / 2.0)

            # clip bounds

            pixel_x_a = max(0, min(pixel_x_a, image_width - 1))
            pixel_y_a = max(0, min(pixel_y_a, image_height - 1))
            pixel_x_b = max(0, min(pixel_x_b, image_width - 1))
            pixel_y_b = max(0, min(pixel_y_b, image_height - 1))
            # print("Pixel coors:",pixel_y_a, pixel_x_a,pixel_y_b, pixel_x_b)
            # print(round(pixel_y_a))
            lum_a = image[int(round(pixel_y_a)), int(round(pixel_x_a))]
            lum_b = image[int(round(pixel_y_b)), int(round(pixel_x_b))]

            diff = lum_a - lum_b
            landmarks_differences.append(diff)
        return landmarks_differences


def compute_pixel_differences(images, sample_feature_locations, landmarks, stage, rotations_inv, shifts_inv):
    # print(landmarks)
    num_landmarks = landmarks.shape[1]
    differences = []
    # print(landmarks.shape)
    # print("Sample features shape:", sample_feature_locations.shape)
    if rotations_inv is None and shifts_inv is None:
        for image, image_landmarks in zip(images, landmarks):
            image_differences = []
            image_height, image_width = image.shape[:2]
            projected_shape = image_landmarks.copy()
            for i in range(num_landmarks):
                image_differences.append(
                    compute_pixel_differences_for_single_image(projected_shape, image, sample_feature_locations,
                                                               image_width, image_height, num_landmarks, stage))
            differences.append(image_differences)
    else:
        for image, image_landmarks, rotation_inv, shift_inv in zip(images, landmarks, rotations_inv, shifts_inv):
            image_differences = []
            image_height, image_width = image.shape[:2]
            projected_shape = project_shape(image_landmarks, rotation_inv, shift_inv)
            for i in range(num_landmarks):
                image_differences.append(
                    compute_pixel_differences_for_single_image(projected_shape, image, sample_feature_locations,
                                                               image_width, image_height, num_landmarks, stage))
            differences.append(image_differences)
    return np.array(differences)


def compute_ground_truth(target_shape, estimated_shape):
    return target_shape - estimated_shape


def create_config_file(config_path, list_of_params):
    with open(config_path, 'w+') as config_file:
        config_file.writelines([str(p) + "\n" for p in list_of_params])
        print(config_path, "created and recorded!")


def create_radii_file(radii_path, radii):
    with open(radii_path, 'w+') as radii_file:
        radii_file.writelines([str(p) + "\n" for p in radii])
        print(radii_path, "created and recorded!")


def parse_config_file(config_path):
    with open(config_path, 'r') as params_file:
        params = []
        params.append(params_file.readline().strip())
        params.append(int(params_file.readline()))
        params.append(int(params_file.readline()))
        params.append(int(params_file.readline()))
        params.append(int(params_file.readline()))
        params.append(int(params_file.readline()))
        params.append(params_file.readline())
    print(params)
    return params


def get_radii_from_file(filename):
    radii = []
    with open(filename, 'r') as params_file:
        radii = [float(l.strip()) for l in params_file.readlines()]
    return radii


if __name__ == "__main__":
    # print(get_sample_feature_locations([0.29, 0.21, 0.16, 0.12, 0.08], 500).shape)
    parse_config_file("ex.txt")
