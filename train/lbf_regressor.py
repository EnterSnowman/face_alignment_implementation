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
                 num_features=500, n_jobs=2, haar_cascade_filename="..\models\haarcascade_frontalface_default.xml",
                 config_file=None, model_name="default_model", trained_models_dir="trained_models"):
        self.forests = []
        self.global_regression_models = []
        self.bin_mappers = []
        self.nums_of_leaves = []
        self.face_detector = cv.CascadeClassifier(haar_cascade_filename)
        if config_file is not None:
            self.config_filename = config_file
            self.params = parse_config_file(config_file)
            self.model_name = self.params[0]
            self.stages = self.params[1]
            self.current_stage = self.params[2]
            self.num_landmarks = self.params[3]
            self.n_trees = self.params[4]
            self.tree_depth = self.params[5]
            self.data_folder = self.params[6]
            self.model_dir = os.path.dirname(config_file)
            print(self.model_dir)
            for s in range(self.current_stage + 1):
                forests, global_regressors = self.load_model_from_stage(
                    os.path.join(self.model_dir, self.model_name + "_stage_" + str(s) + ".pkl"))
                stage_bin_mappers = self.load_bin_mappers_from_stage(os.path.join(self.model_dir, self.model_name +
                                                                                  "_bin_feature_mapper_stage_" + str(
                    s) + ".pkl"))
                stage_nums_of_leaves = self.load_num_of_leaves(os.path.join(self.model_dir, self.model_name +
                                                                            "_nums_of_leaves_stage_" + str(s) + ".pkl"))
                self.forests.append(forests)
                self.global_regression_models.append(global_regressors)
                self.bin_mappers.append(stage_bin_mappers)
                self.nums_of_leaves.append(stage_nums_of_leaves)
            self.sampled_feature_locations = self.load_sample_feature_locations(
                os.path.join(self.model_dir, self.model_name + "_sample_feature_locations.pkl"))
            self.radii = get_radii_from_file(os.path.join(self.model_dir, self.model_name + "_radii.txt"))
            self.is_trained_before = True
        else:
            self.trained_models_dir = trained_models_dir
            self.is_trained_before = False
            self.stages = len(radii)
            self.n_trees = n_trees
            self.num_landmarks = num_landmarks
            self.sampled_feature_locations = get_sample_feature_locations(radii, num_features)
            self.current_stage = -1
            self.tree_depth = tree_depth
            self.model_name = model_name
            self.radii = radii

    def load_data(self, data_folder, is_debug=False, debug_size=20, image_format=".jpg"):
        self.data_folder = data_folder
        ims, lms, bounding_boxes = get_images_with_bounding_boxes(data_folder, is_debug=is_debug, debug_size=debug_size,
                                                                  image_format=image_format)
        print("Landmarks shape:", lms.shape)
        print("Images len:", len(ims))

        self.mean_shape, localized_landmarks = get_mean_shape_and_localized_landmarks(lms, bounding_boxes)
        # self.images, localized_landmarks, self.normalized_shapes, rotations, self.rotations_inv, shifts, \
        # self.shifts_inv, pupil_distances, self.estimated_shapes = get_training_data(ims, localized_landmarks,
        #                                                                             bounding_boxes, self.mean_shape)
        self.rotations_inv, self.shifts_inv = None, None
        self.images, self.normalized_shapes, self.estimated_shapes = get_training_data_without_normalization(ims,
                                                                                                             localized_landmarks,
                                                                                                             bounding_boxes,
                                                                                                             self.mean_shape)
        if self.is_trained_before:
            self.estimated_shapes = self.load_estimated_shapes(
                os.path.join(self.model_dir, self.model_name + "_current_estimated_shapes.pkl"))
        if self.current_stage + 1 < self.stages:
            self.pixel_differences = compute_pixel_differences(self.images, self.sampled_feature_locations,
                                                               self.estimated_shapes, self.current_stage + 1,
                                                               self.rotations_inv, self.shifts_inv)

        self.ground_truth = compute_ground_truth(self.normalized_shapes, self.estimated_shapes)
        # print("Pix diff size", self.pixel_differences.shape)
        # print("Target shape size:", self.normalized_shapes.shape)
        # print("Est shape size:", self.estimated_shapes.shape)
        # print("Ground truth shape size:", self.ground_truth.shape)

    def load_model_from_stage(self, stage_filename):
        with open(stage_filename, 'rb') as f:
            forests_stage, global_linear_regressor_stage = pickle.load(f)
        return forests_stage, global_linear_regressor_stage

    def load_bin_mappers_from_stage(self, bin_mapper_filename):
        with open(bin_mapper_filename, 'rb') as f:
            stage_bin_mappers = pickle.load(f)
        return stage_bin_mappers

    def load_sample_feature_locations(self, sample_features_locations_filename):
        with open(sample_features_locations_filename, 'rb') as f:
            sample_features_locations = pickle.load(f)
        return sample_features_locations

    def load_estimated_shapes(self, estimated_shapes_filename):
        with open(estimated_shapes_filename, 'rb') as f:
            estimated_shapes = pickle.load(f)
        return estimated_shapes

    def load_num_of_leaves(self, nums_of_leaves_filename):
        with open(nums_of_leaves_filename, 'rb') as f:
            nums_of_leaves = pickle.load(f)
        return nums_of_leaves

    def save_sample_feature_locations(self):
        filename = os.path.join(self.model_dir, self.model_name + "_sample_feature_locations.pkl")
        if ~os.path.exists(filename):
            with open(filename, 'w+'):
                print(filename, "created")
        with open(filename, 'wb') as pickle_file:
            pickle.dump(self.sampled_feature_locations, pickle_file)
            print("Sample feature locations saved!")

    def train(self):
        print("Start training")
        if not self.is_trained_before:
            if not os.path.exists(os.path.join(self.trained_models_dir, self.model_name)):
                new_model_folder = os.path.join(self.trained_models_dir, self.model_name)
                os.makedirs(new_model_folder)
                self.config_filename = os.path.join(new_model_folder, self.model_name + "_conf.txt")
                create_config_file(self.config_filename, [self.model_name, self.stages, -1, self.num_landmarks,
                                                          self.n_trees, self.tree_depth, self.data_folder])
                create_radii_file(os.path.join(new_model_folder, self.model_name + "_radii.txt"), self.radii)
                self.model_dir = new_model_folder
                self.save_sample_feature_locations()
            else:
                print("Folder with this model name exists. Please, use another name.")
                return -1
        for stage in range(self.current_stage + 1, self.stages):
            print("Stage", stage + 1)
            self.train_forests(stage)
            self.binary_features = self.get_binary_features(stage)
            self.train_global_linear_regression(stage)
            self.update_data(stage)
            self.save_stage(stage)
            save_last_completed_stage_to_config_file(self.config_filename, stage)

    def train_forests(self, stage):
        print("Train forests in stage:", stage + 1)
        stage_forests = []
        for landmark_index in range(self.num_landmarks):
            stage_forests.append(RandomForestRegressor(n_estimators=self.n_trees, max_depth=self.tree_depth, n_jobs=-1))
        stage_binary_mappers = []
        stage_nums_of_leaves = []
        self.forests.append(stage_forests)
        for landmark_index in range(self.num_landmarks):
            landmark_binary_mappers = []
            landmark_nums_leaves = []
            print("train forest for landmark #", landmark_index)
            start = time.time()
            self.forests[stage][landmark_index].fit(self.pixel_differences[:, landmark_index, :],
                                                    self.ground_truth[:, landmark_index, :])
            print("Train time:", time.time() - start)
            for tree in self.forests[stage][landmark_index].estimators_:
                node_to_leaf_dict, num_of_leaves = get_dict_node_to_leaf_number(tree)
                landmark_nums_leaves.append(num_of_leaves)
                landmark_binary_mappers.append(node_to_leaf_dict)
            stage_binary_mappers.append(landmark_binary_mappers)
            stage_nums_of_leaves.append(landmark_nums_leaves)
        self.bin_mappers.append(stage_binary_mappers)
        self.nums_of_leaves.append(stage_nums_of_leaves)

    def train_global_linear_regression(self, stage):
        print("Train global regression in stage", stage + 1)
        stage_global_regression_models = []
        for landmark_index in range(self.ground_truth.shape[1]):
            stage_global_regression_models.append(
                [LinearSVR(C=0.00001, epsilon=0., loss='squared_epsilon_insensitive'),
                 LinearSVR(C=0.00001, epsilon=0., loss='squared_epsilon_insensitive')])
        self.global_regression_models.append(stage_global_regression_models)
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
                    binary_index = self.bin_mappers[stage][j][tree_index][leaves_indices[tree_index]]
                    row_ind.append(i)
                    col_ind.append(forest_pointer + pointer + binary_index)
                    data.append(1)
                    pointer += self.nums_of_leaves[stage][j][tree_index]
            forest_pointer += pointer
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
            for tree_index in range(len(self.forests[stage][j].estimators_)):
                # node_to_leaf_dict, num_of_leaves = get_dict_node_to_leaf_number(tree)
                binary_index = self.bin_mappers[stage][j][tree_index][leaves_indices[0, tree_index]]
                data.append(1)
                row_ind.append(0)
                col_ind.append(pointer + binary_index)
                pointer += self.nums_of_leaves[stage][j][tree_index]
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

    def save_stage(self, stage):
        # save forests and SVRs
        filename = os.path.join(self.model_dir, self.model_name + "_stage_" + str(stage) + ".pkl")
        if ~os.path.exists(filename):
            with open(filename, 'w+'):
                print(filename, "created")
        with open(filename, 'wb') as pickle_file:
            pickle.dump([self.forests[stage], self.global_regression_models[stage]], pickle_file)
            print("Model at stage", stage, "saved!")

        # save estimated shapes
        est_shapes_filename = os.path.join(self.model_dir, self.model_name + "_current_estimated_shapes.pkl")
        if ~os.path.exists(est_shapes_filename):
            with open(est_shapes_filename, 'w+'):
                print(est_shapes_filename, "created")
        with open(est_shapes_filename, 'wb') as pickle_file:
            pickle.dump(self.estimated_shapes, pickle_file)
            print("Estimated shapes at stage", stage, "saved!")
        # save bin mappers

        mapper_filename = os.path.join(self.model_dir,
                                       self.model_name + "_bin_feature_mapper_stage_" + str(stage) + ".pkl")
        if ~os.path.exists(mapper_filename):
            with open(mapper_filename, 'w+'):
                print(mapper_filename, "created")
        with open(mapper_filename, 'wb') as pickle_file:
            pickle.dump(self.bin_mappers[stage], pickle_file)
            print("Mapper at stage", stage, "saved!")

        leaves_filename = os.path.join(self.model_dir, self.model_name + "_nums_of_leaves_stage_" + str(stage) + ".pkl")
        if ~os.path.exists(leaves_filename):
            with open(leaves_filename, 'w+'):
                print(leaves_filename, "created")
        with open(leaves_filename, 'wb') as pickle_file:
            pickle.dump(self.nums_of_leaves[stage], pickle_file)
            print("Leaves at stage", stage, "saved!")
        # save nums of leaves

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
            fw = w / 2
            fh = h / 2
            estimated_shape = self.mean_shape.copy()
            for p in estimated_shape:
                cv.circle(roi_color, (int(p[0] * fw + fw), int(p[1] * fh + fh)), 4, (255, 0, 0), -1)
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
