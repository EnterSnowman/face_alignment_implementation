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

            diff = int(lum_a) - int(lum_b)
            landmarks_differences.append(diff)
        return landmarks_differences


def get_similarity_transform(shape_from, shape_to):
    rotation = np.zeros((2, 2))
    scale = 0.
    center_x_1, center_y_1 = np.mean(shape_to, axis=0)
    center_x_2, center_y_2 = np.mean(shape_from, axis=0)

    center_x_1 /= shape_to.shape[0]
    center_y_1 /= shape_to.shape[0]
    center_x_2 /= shape_from.shape[0]
    center_y_2 /= shape_from.shape[0]

    temp1 = shape_to.copy()
    temp2 = shape_from.copy()

    temp1 -= np.array([center_x_1, center_y_1]).reshape((1, 2))
    temp2 -= np.array([center_x_2, center_y_2]).reshape((1, 2))

    covariance1, mean1 = cv.calcCovarMatrix(temp1, None, cv.COVAR_COLS)
    # print("cov mean:", covariance1, mean1)
    covariance2, mean2 = cv.calcCovarMatrix(temp2, None, cv.COVAR_COLS)

    s1, s2 = np.sqrt(np.linalg.norm(covariance1)), np.sqrt(np.linalg.norm(covariance2))

    scale = s1 / s2
    temp1 /= s1
    temp2 /= s2

    num = 0.
    den = 0.

    for t1, t2 in zip(temp1, temp2):
        num += t1[1] * t2[0] - t1[0] * t2[1]
        den += t1[0] * t2[0] + t1[1] * t2[1]

    norm = np.sqrt(num * num + den * den)
    sin_theta = num / norm
    cos_theta = den / norm
    rotation[0, 0] = cos_theta
    rotation[0, 1] = -sin_theta
    rotation[1, 0] = sin_theta
    rotation[1, 1] = cos_theta
    # print("rot scale", rotation, scale)
    return rotation, scale


def compute_pixel_differences_at_stage(images, sample_feature_locations, landmarks, stage, rotations, scales, bounding_boxes):
    pixel_differences = []
    num_landmarks = landmarks.shape[1]
    for image, image_landmark, rotation, scale, bounding_box in zip(images, landmarks, rotations, scales,
                                                                    bounding_boxes):
        image_pixel_differences = []
        for i in range(num_landmarks):
            landmark_differences = []
            landmark_x, landmark_y = image_landmark[i, 0], image_landmark[i, 1]
            for sample_feature_location in sample_feature_locations[stage, :, :]:
                delta_x = rotation[0, 0] * sample_feature_location[0] + rotation[0, 1] * sample_feature_location[1]
                delta_y = rotation[1, 0] * sample_feature_location[0] + rotation[1, 1] * sample_feature_location[1]
                delta_x = scale * delta_x * bounding_box[2] / 2.0
                delta_y = scale * delta_y * bounding_box[3] / 2.0
                real_x = delta_x + landmark_x
                real_y = delta_y + landmark_y
                real_x = max(0, min(real_x, image.shape[1] - 1))
                real_y = max(0, min(real_y, image.shape[0] - 1))
                tmp = int(image[int(round(real_y)), int(round(real_x))])

                delta_x = rotation[0, 0] * sample_feature_location[2] + rotation[0, 1] * sample_feature_location[3]
                delta_y = rotation[1, 0] * sample_feature_location[2] + rotation[1, 1] * sample_feature_location[3]
                delta_x = scale * delta_x * bounding_box[2] / 2.0
                delta_y = scale * delta_y * bounding_box[3] / 2.0
                real_x = delta_x + landmark_x
                real_y = delta_y + landmark_y
                real_x = max(0, min(real_x, image.shape[1] - 1))
                real_y = max(0, min(real_y, image.shape[0] - 1))

                landmark_differences.append(tmp - int(image[int(round(real_y)), int(round(real_x))]))
            image_pixel_differences.append(landmark_differences)
        pixel_differences.append(image_pixel_differences)
    res = np.array(pixel_differences)
    print(res.shape)
    return res



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
        params.append(int(params_file.readline().strip()))
        params.append(int(params_file.readline().strip()))
        params.append(int(params_file.readline().strip()))
        params.append(int(params_file.readline().strip()))
        params.append(int(params_file.readline().strip()))
        params.append(params_file.readline().strip())
        params.append(params_file.readline().strip())  # rf type
    print(params)
    return params


def save_last_completed_stage_to_config_file(config_path, stage):
    with open(config_path, 'r') as params_file:
        params = [l.strip() for l in params_file.readlines()]
    params[2] = stage
    with open(config_path, 'w') as params_file:
        params_file.writelines([str(p) + "\n" for p in params])


def get_radii_from_file(filename):
    radii = []
    with open(filename, 'r') as params_file:
        radii = [float(l.strip()) for l in params_file.readlines()]
    return radii


if __name__ == "__main__":
    print(get_sample_feature_locations([0.29, 0.21, 0.16, 0.12, 0.08], 500).shape)
    # parse_config_file("ex.txt")
