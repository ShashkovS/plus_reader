import plus_reader as pr
import unittest
import os

# TODO: Сделать нормальное количество нормальных тестов, в том числе отдельных функций

pr.VERBOSE = True
os.chdir('test_imgs&pdfs')


class TestTableRecognition(unittest.TestCase):
    def total_test_1(self):
        print('FOO')
        images = pr.extract_images_from_pdf('tst_01.pdf', pages_to_process=[0, 1])
        recognized_pages = pr.prc_all_images(images)
        self.assertEqual(len(recognized_pages), 2)
        print(recognized_pages)


if __name__ == '__main__':
    unittest.main()