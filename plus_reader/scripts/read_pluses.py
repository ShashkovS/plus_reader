#!/usr/bin/env python
import argparse
import sys



def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description='Reads tables of pluses or other marks on scans (pdf or images)'
    )
    # TODO: Сделать нормальные параметры
    parser.add_argument('image', nargs='+')
    parser.add_argument('pdf', nargs='+')
    parser.add_argument('dest', nargs='+')
    args = parser.parse_args(args)

    # TODO: Сделать вызов распознавателя

if __name__ == '__main__':
    main()