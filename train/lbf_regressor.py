from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from utils import *
from preprocessing_loading.preprocessor import get_images_with_bounding_boxes, get_mean_shape_and_localized_landmarks, \
    get_training_data, get_training_data_without_normalization
from scipy import sparse
from utils.bounding_box import make_square_bounding_box, expand_bounding_box
import pickle
import os
import time
import cv2 as cv


class LBFRegressor:

    def __init__(self, tree_depth=7, n_trees=1200, radii=[0.29, 0.21, 0.16, 0.12, 0.08], num_landmarks=68,
                 num_features=500, n_jobs=2, haar_cascade_filename="..\models\haarcascade_frontalface_default.xml"):
        self.face_detector = cv.CascadeClassifier(haar_cascade_filename)
        self.stages = len(radii)
        self.forests = []
        self.n_trees = n_trees
        self.num_landmarks = num_landmarks
        self.sampled_feature_locations = get_sample_feature_locations(radii, num_features)
        self.global_regression_models = []
        for i in range(self.stages):
            stage_forests = []
            stage_global_regression_models = []
            for j in range(num_landmarks):
                stage_forests.append(RandomForestRegressor(n_estimators=n_trees, max_depth=tree_depth, n_jobs=n_jobs))
                # [x, y]
                stage_global_regression_models.append(
                    [LinearSVR(C=0.00001, epsilon=0., loss='squared_epsilon_insensitive'),
                     LinearSVR(C=0.00001, epsilon=0., loss='squared_epsilon_insensitive')])
            self.forests.append(stage_forests)
            self.global_regression_models.append(stage_global_regression_models)

    def load_data(self, data_folder, is_debug=False, debug_size=20, image_format=".jpg"):
        ims, lms, bounding_boxes = get_images_with_bounding_boxes(data_folder, is_debug=is_debug, debug_size=debug_size,
                                                                  image_format=image_format)
        print("Landmarks shape:", lms.shape)
        print("Images len:", len(ims))

        self.mean_shape, localized_landmarks = get_mean_shape_and_localized_landmarks(lms, bounding_boxes)
        # self.images, localized_landmarks, self.normalized_shapes, rotations, self.rotations_inv, shifts, \
        # self.shifts_inv, pupil_distances, self.estimated_shapes = get_training_data(ims, localized_landmarks,
        #                                                                             bounding_boxes, self.mean_shape)
        self.rotations_inv, self.shifts_inv = None, None
        self.images, self.normalized_shapes, self.estimated_shapes = get_training_data_without_normalization(ims, localized_landmarks, bounding_boxes, self.mean_shape)
        self.pixel_differences = compute_pixel_differences(self.images, self.sampled_feature_locations,
                                                           self.estimated_shapes, 0,
                                                           self.rotations_inv, self.shifts_inv)

        self.ground_truth = compute_ground_truth(self.normalized_shapes, self.estimated_shapes)
        print("Pix diff size", self.pixel_differences.shape)
        print("Target shape size:", self.normalized_shapes.shape)
        print("Est shape size:", self.estimated_shapes.shape)
        print("Ground truth shape size:", self.ground_truth.shape)

    def train(self):
        print("Start training")
        for stage in range(self.stages):
            print("Stage", stage + 1)
            self.train_forests(stage)
            self.binary_features = self.get_binary_features(stage)
            self.train_global_linear_regression(stage)
            self.update_data(stage)

    def train_forests(self, stage):
        print("Train forests in stage:", stage + 1)
        for landmark_index in range(self.num_landmarks):
            print("train forest for landmark #", landmark_index)
            start = time.time()
            self.forests[stage][landmark_index].fit(self.pixel_differences[:, landmark_index, :],
                                                    self.ground_truth[:, landmark_index, :])
            print("Train time:", time.time() - start)

    def train_global_linear_regression(self, stage):
        print("Train global regression in stage", stage + 1)
        for landmark_index in range(self.ground_truth.shape[1]):
            for coor in range(self.ground_truth.shape[2]):
                print("Train global regression for landmark #", landmark_index, "coor", coor)
                start = time.time()
                self.global_regression_models[stage][landmark_index][coor].fit(self.binary_features,
                                                                               self.ground_truth[:, landmark_index,
                                                                               coor])

                print("Train time:", time.time() - start)

    def get_binary_features(self, stage):
        # binary_features = []
        row_ind = []
        col_ind = []
        data = []
        tree_maps = []
        nums_of_leaves = []
        for j in range(self.num_landmarks):
            single_forest_maps = []
            forest_nums_of_leaves = []
            for tree in self.forests[stage][j].estimators_:
                node_to_leaf_dict, num_of_leaves = get_dict_node_to_leaf_number(tree)
                single_forest_maps.append(node_to_leaf_dict)
                forest_nums_of_leaves.append(num_of_leaves)
            tree_maps.append(single_forest_maps)
            nums_of_leaves.append(forest_nums_of_leaves)
        # all_leaves_indices
        forest_pointer = 0
        for j in range(self.num_landmarks):
            print("Calc bin features for forest", j)
            landmarks_indices = self.forests[stage][j].apply(self.pixel_differences[:, j, :])
            pointer = 0
            for i in range(self.pixel_differences.shape[0]):
                pointer = 0
                leaves_indices = landmarks_indices[i, :]
                for tree_index in range(self.n_trees):
                    binary_index = tree_maps[j][tree_index][leaves_indices[tree_index]]
                    # print(i)
                    row_ind.append(i)
                    col_ind.append(forest_pointer + pointer + binary_index)
                    data.append(1)
                    pointer += nums_of_leaves[j][tree_index]
            forest_pointer += pointer

        # for i in range(self.pixel_differences.shape[0]):
        #     pointer = 0
        #     print("Calc bin features for example", i)
        #     for j in range(self.num_landmarks):
        #         leaves_indices = self.forests[stage][j].apply(self.pixel_differences[i, j, :].reshape(1, -1))
        #         for tree_index in range(self.n_trees):
        #             binary_index = tree_maps[j][tree_index][leaves_indices[0, tree_index]]
        #             row_ind.append(i)
        #             col_ind.append(pointer + binary_index)
        #             data.append(1)
        #             pointer += nums_of_leaves[j][tree_index]
        return sparse.coo_matrix((data, (row_ind, col_ind)))

    def get_binary_features_for_single_image(self, stage, image, estimated_shape):
        diff = compute_pixel_differences([image], self.sampled_feature_locations, np.array([estimated_shape]), stage,
                                         None, None)
        data = []
        row_ind = []
        col_ind = []
        pointer = 0
        # print(diff.shape)
        for j in range(self.num_landmarks):
            leaves_indices = self.forests[stage][j].apply(diff[0, j, :].reshape(1, -1))
            for tree_index, tree in enumerate(self.forests[stage][j].estimators_):
                node_to_leaf_dict, num_of_leaves = get_dict_node_to_leaf_number(tree)
                binary_index = node_to_leaf_dict[leaves_indices[0, tree_index]]
                data.append(1)
                row_ind.append(0)
                col_ind.append(pointer + binary_index)
                pointer += num_of_leaves
        # make correct shape
        data.append(0)
        row_ind.append(0)
        col_ind.append(pointer - 1)
        # print("pointer:", pointer)
        return sparse.coo_matrix((data, (row_ind, col_ind)))

    def update_data(self, stage):
        for i in range(self.estimated_shapes.shape[1]):
            for j in range(self.estimated_shapes.shape[2]):
                delta = self.global_regression_models[stage][i][j].predict(self.binary_features)
                self.estimated_shapes[:, i, j] += delta
        if stage < self.stages - 1:
            self.pixel_differences = compute_pixel_differences(self.images, self.sampled_feature_locations,
                                                               self.estimated_shapes, stage + 1,
                                                               self.rotations_inv, self.shifts_inv)
        self.ground_truth = compute_ground_truth(self.normalized_shapes, self.estimated_shapes)

    def get_delta_for_image_from_global_regression(self, stage, image_binary_features):
        delta = []
        for i in range(len(self.global_regression_models[stage])):
            d_landmark = []
            for j in range(len(self.global_regression_models[stage][0])):
                d = self.global_regression_models[stage][i][j].predict(image_binary_features)
                d_landmark.append(d)
            delta.append(d_landmark)
        return np.array(delta)

    def save_model(self, filename):
        if ~os.path.exists(filename):
            with open(filename, 'w+'):
                print(filename, "created")
        with open(filename, 'wb') as pickle_file:
            pickle.dump([self.forests, self.global_regression_models, self.sampled_feature_locations, self.mean_shape],
                        pickle_file)
            print("model saved!")

    def predict(self, image):
        image_height = image.shape[0]
        image_width = image.shape[1]
        # bgr to gray
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # bounding box
        faces = self.face_detector.detectMultiScale(gray, 1.2, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]

            x, y, w, h = make_square_bounding_box([x, y, w, h], image_height, image_width)
            x, y, w, h = expand_bounding_box([x, y, w, h], image_width, image_height)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = image[y:y + h, x:x + w]
            # put mean shape
            # get pixel diff stage 1
            # pass through models

            estimated_shape = self.mean_shape.copy()

            for stage in range(self.stages):
                print("Stage", stage + 1)
                start = time.time()
                image_binary_features = self.get_binary_features_for_single_image(stage, roi_gray, estimated_shape)
                print("Calc bin features:", time.time() - start)
                start = time.time()
                image_delta = self.get_delta_for_image_from_global_regression(stage, image_binary_features)
                print("Get delta from global :", time.time() - start)
                # print("image delta:", image_delta)
                estimated_shape += image_delta.reshape(estimated_shape.shape)
            # print("Predict time:", time.time() - start)
            fw = w / 2
            fh = h / 2
            for p in estimated_shape:
                cv.circle(roi_color, (int(p[0] * fw + fw), int(p[1] * fh + fh)), 4, (0, 0, 255), -1)
            cv.imshow(str(time.time()), roi_color)
            cv.waitKey(0)
            cv.destroyAllWindows()
            return estimated_shape
        else:
            print("No faces!")
            return None

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            self.forests = model[0]
            self.global_regression_models = model[1]
            self.sampled_feature_locations = model[2]
            self.mean_shape = model[3]
            print("model loaded!")


if __name__ == "__main__":
    model_filename = "..\\trained_models\model_14_my_photos.pkl"
    folder = "..\data\my_photos"
    n_landmarks = 83
    d = 5
    n_trees = 300
    model = LBFRegressor(num_landmarks=n_landmarks, n_trees=n_trees)
    model.load_data(folder, is_debug=False, debug_size=2, image_format=".png")
    model.train()
    model.save_model(model_filename)
    # model2 = LBFRegressor()
    # model2.load_model("..\\trained_models\model_1.pkl")
    # model.print_tree_type()
    # print(compute_pixel_differences(images, model.sampled_feature_locations, estimated_shapes, 0, rotations_inv, shifts_inv).shape)
    # print(compute_ground_truth(normalized_shapes, estimated_shapes))
    # model.train(list_of_images, landmark_positions)
