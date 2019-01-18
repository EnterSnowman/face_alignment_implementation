from train.lbf_regressor import LBFRegressor
import glob
import os
import cv2 as cv


def show_predicted_landmarks_on_test_photos(config_path, test_photos_folder, data_folder):
    model = LBFRegressor(haar_cascade_filename="..\models\haarcascade_frontalface_default.xml", num_landmarks=14,
                         config_file=config_path)
    model.load_data(data_folder, image_format=".png")
    pathes = []
    estimated_shapes = []
    for image_path in glob.glob(os.path.join(test_photos_folder, "*")):
        image = cv.imread(image_path)
        est_shape = model.predict(image)
        if est_shape is not None:
            pathes.append(image_path)
            estimated_shapes.append(est_shape)
            print(image_path)
            print(est_shape)


if __name__ == "__main__":
    show_predicted_landmarks_on_test_photos("..\\trained_models\model_14_300_5\model_14_300_5_conf.txt",
                                            "..\data\st2", "..\data\my_photos_14")
