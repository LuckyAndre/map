import numpy as np
import cv2

from map_utils import homography_matrix, make_grid, make_map, make_contour, draw_parking

# области парковок
CENTRAL_PARKING = [(280, 90), (955, 90), (955, 150), (280, 150), (280, 90)]
BOTTOM_PARKING = [(290, 220), (920, 185), (970, 195), (965, 220), (290, 255), (290, 220)]
TOP_PARKING = [(600, 10), (940, 10), (940, 60), (600, 60), (600, 10)]
LEFT_PARKING = [(0, 140), (300, 150), (300, 200), (0, 190), (0, 140)]
RIGHT_PARKING = [(950, 20), (1020, 80), (1020, 200), (980, 200), (980, 120), (950, 70), (950, 20)]


def create_map(bbox_tensor_lst: list):

    # получаю матрицу проективного преобразования
    transform_mat, status = homography_matrix()

    # делаю проекцию всех bbox
    bbox_mapped_lst = []
    for bbox in bbox_tensor_lst:
        car_center_source = bbox.numpy()[0][0:2]
        car_center_target = cv2.perspectiveTransform(car_center_source[None, None, :].astype(np.float32),
                                                     transform_mat).ravel()
        bbox_mapped_lst.append(car_center_target)

    # сетка для спроецированного изображения (для каждого bbox будет определяться - к какой линии он ближе)
    grids_source_lst = make_grid()

    # создаю шаблон карты-схемы
    parking_map, grids_target_lst = make_map(len(grids_source_lst))

    # верхняя парковка
    contour = make_contour(TOP_PARKING)
    parking_map, cap_top = draw_parking(bbox_mapped_lst, parking_map, contour, grids_source_lst, grids_target_lst,
                          y_level=30, pos=0, capacity=12)

    # правая парковка
    contour = make_contour(RIGHT_PARKING)
    parking_map, cap_right = draw_parking(bbox_mapped_lst, parking_map, contour, grids_source_lst, grids_target_lst,
                          y_level=None, pos=0, capacity=4)

    # нижняя парковка
    contour = make_contour(BOTTOM_PARKING)
    parking_map, cap_bot = draw_parking(bbox_mapped_lst, parking_map, contour, grids_source_lst, grids_target_lst,
                          y_level=230, pos=90, capacity=10)

    # левая парковка
    contour = make_contour(LEFT_PARKING)
    parking_map, cap_left = draw_parking(bbox_mapped_lst, parking_map, contour, grids_source_lst, grids_target_lst,
                          y_level=170, pos=125, capacity=9)

    # центральная парковка
    contour = make_contour(CENTRAL_PARKING)
    parking_map, cap_cen = draw_parking(bbox_mapped_lst, parking_map, contour, grids_source_lst, grids_target_lst,
                          y_level=120, pos=0, capacity=22)

    return parking_map, {'top': cap_top, 'right': cap_right, 'bottom': cap_bot, 'left': cap_left, 'central': cap_cen}

