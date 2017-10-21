# -*- coding: utf-8 -*-.
import numpy as np

# TODO: убрать магические константы
MIN_FILLED_PART = 0.030
BLACK_DOT_THRESHOLD = 225
MIN_FILLED_DOTS = 1000

def recognize_cell(np_cell_image):
    part = 1 - (np.sum(np_cell_image) / np_cell_image.size / 255)
    return (part >= MIN_FILLED_PART or np_cell_image[np_cell_image < BLACK_DOT_THRESHOLD].size >= MIN_FILLED_DOTS)


if __name__ == '__main__':
    pass