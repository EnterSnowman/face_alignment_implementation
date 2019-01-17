import numpy as np
import cv2 as cv
import os


def get_bounding_boxes(list_of_image_names, landmarks, make_square=True):
    bounding_boxes = []
    for image_name, image_landmarks in zip(list_of_image_names, landmarks):
        image_rgb = cv.imread(image_name)
        bounding_boxes.append(get_single_bounding_box(image_name, image_rgb, image_landmarks, make_square))
    return bounding_boxes


def get_single_bounding_box(image_name, image, image_landmarks, make_square=True):
    image_height = image.shape[0]
    image_width = image.shape[1]
    # print("Im h:", image_height, "Im w:", image_width)
    # print(image_landmarks)

    (x, y, w, h) = cv.boundingRect(image_landmarks.astype(int))
    # print("Bbox before squaring:", x, y, w, h)
    if make_square:
        x, y, w, h = make_square_bounding_box([x, y, w, h], image_height, image_width)

    if x < 0 or y < 0 or x + w > image_width or y + h > image_height:
        print("Unable to make correct square!!!")
        os.remove(image_name)
        print(image_name, "removed!")
    return expand_bounding_box([x, y, w, h], image_width, image_height)


def expand_bounding_box(box, image_width, image_height):
    padding = box[2] * 0.3
    padding = min(padding, box[0])
    padding = min(padding, box[1])
    padding = min(padding, image_width - (box[0] + box[2]))
    padding = min(padding, image_height - (box[1] + box[3]))
    box[0] -= int(padding)
    box[1] -= int(padding)
    box[2] += int(2 * padding)
    box[3] += int(2 * padding)
    return box[0], box[1], box[2], box[3]


def make_square_bounding_box(box, image_height, image_width):
    # print("Im h:", image_height, "Im w:", image_width)
    # print("Bbox:", box)
    x, y, w, h = box[0], box[1], box[2], box[3]
    bbox_width = box[2]
    bbox_height = box[3]
    top = y
    bottom = y + bbox_height
    left = x
    right = x + bbox_width

    if bbox_width > bbox_height:
        diff = (bbox_width - bbox_height) // 2
        mod = (bbox_width - bbox_height) % 2
        top = top - diff
        bottom = bottom + diff + mod
        if top < 0:
            bottom -= top
            top -= top

            # bbox.move_y(-bbox.top)
            # assert bottom <= image_height
        elif bottom > image_height:
            top += image_height - bottom
            bottom += image_height - bottom

            # assert top >= 0

    elif bbox_width < bbox_height:
        diff = (bbox_height - bbox_width) // 2
        mod = (bbox_height - bbox_width) % 2
        left = left - diff
        right = right + diff + mod
        if left < 0:
            right -= left
            left -= left

            # assert right <= image_width
        elif right > image_width:
            left += image_width - right
            right += image_width - right
            # bbox.move_x(image_width - right)
            # assert left >= 0
    # print("Bbox after squaring:", left, top, right - left, bottom - top)
    assert right - left == bottom - top
    return left, top, right - left, bottom - top
