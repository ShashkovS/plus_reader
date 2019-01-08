# -*- coding: utf-8 -*-.
import numpy as np

# TODO: убрать магические константы
MIN_FILLED_PART = 0.030
BLACK_DOT_THRESHOLD = 225
MIN_FILLED_DOTS = 1000


def recognize_cell(np_cell_image):
    part = 1 - (np.sum(np_cell_image) / np_cell_image.size / 255)
    return (part >= MIN_FILLED_PART or np_cell_image[np_cell_image < BLACK_DOT_THRESHOLD].size >= MIN_FILLED_DOTS)


def ext_find_filled_cells(gray_np_image, coords_of_horiz_lns, coords_of_vert_lns):
    """Самая главная функция — определяет, заполнена ли ячейка
    """
    if not coords_of_vert_lns or not coords_of_horiz_lns:
        return np.zeros((1, 1), np.bool)
    rows, colums = len(coords_of_horiz_lns) - 1, len(coords_of_vert_lns) - 1
    filled = np.zeros((rows, colums), np.bool)
    for i in range(rows):
        for j in range(colums):
            block = gray_np_image[coords_of_horiz_lns[i]:coords_of_horiz_lns[i + 1], coords_of_vert_lns[j]:coords_of_vert_lns[j + 1]]
            filled[i][j] = bool(recognize_cell(block))
    return filled


if __name__ == '__main__':
    pass