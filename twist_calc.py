#!/usr/bin/env python

from single_point_SF import main as sfr_main
from single_point_Energy import main as energy_main

from utility import print_git_commit
import argparse

if __name__ == '__main__':
    print_git_commit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int)
    args = parser.parse_args()
    number = args.number
    sfr_main(f'{number}.xyz', cuda_device = 'cpu', easy_delta_E = True, approximation = 'overlap', save_name = f'sfr_{number}')
    energy_main(f'{number}.xyz', cuda_device = 'cpu', save_name = f'energy_{number}')

    print('Single point program finished!')