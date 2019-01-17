import cv2 as cv
import imutils
from preprocessing_loading.loader import get_list_of_images_and_landmarks
from utils.bounding_box import get_bounding_boxes


def show_photos(list_of_images, landmarks, bounding_boxes):
    current_file = 0
    show_single_photo(list_of_images[current_file], landmarks[current_file, :, :], bounding_boxes[current_file])
    while True:
        k = cv.waitKey(1)
        is_press = False
        if k == 100:
            current_file += 1
            is_press = True
        if k == 97:
            current_file -= 1
            is_press = True
        if is_press:
            show_single_photo(list_of_images[current_file], landmarks[current_file, :, :], bounding_boxes[current_file])
        if k == 27:
            break


def show_single_photo(image_name, image_landmarks, bounding_box):
    photo = cv.imread(image_name)

    for coor in image_landmarks:
        cv.circle(photo, (int(coor[0]), int(coor[1])), 3, (0, 0, 255), -1)
    print(bounding_box)
    cv.rectangle(photo, (bounding_box[0], bounding_box[1]),
                 (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (0, 255, 0), 3)
    photo = imutils.resize(photo, height=800)
    # photo = cv.resize(photo, (750, 750))
    cv.imshow("test", photo)


if __name__ == "__main__":
    ims, lms = get_list_of_images_and_landmarks("..\data\helen\\trainset")
    bounding_boxes = get_bounding_boxes(ims, lms)
    show_photos(ims, lms, bounding_boxes)
