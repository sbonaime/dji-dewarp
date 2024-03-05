#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import numpy as np
from exiftool import ExifToolHelper, ExifTool

RED = "\033[91m"
END = "\033[0m"


def is_valid_directory(arg):
    if not os.path.isdir(arg):
        print(RED, "\"" + arg + "\"""  is not a valid directory" + END)
        sys.exit()
    return arg


def main():
    parser = argparse.ArgumentParser(
        description="Dewarp PH4RTK with exif  dewarpdata.")
    parser.add_argument("input_dir", type=is_valid_directory,
                        help="Input directory")
    parser.add_argument(
        "output_dir", type=is_valid_directory, help="Outpu directory")
    parser.add_argument("-e", "--intput_fil_ext",
                        help="file extension to use", default='JPG', required=False)

    args = parser.parse_args()

    # Get  arguments
    intput_fil_ext = args.intput_fil_ext
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    # Select files with in
    for file in os.listdir(input_dir):
        file = os.path.join(input_dir, file)
        if file.endswith(intput_fil_ext):
            print(f"[*] Read : {file}")
            input_image = cv2.imread(file)
            print(f"\t[*] Dewarp : {file}")

            # Get Dewarp coeficients
            with ExifToolHelper()as et:
                for d in et.get_metadata(file):
                    for tag, value in d.items():
                        if 'XMP:DewarpData' in tag:
                            fx, fy, cx, cy, k1, k2, p1, p2, k3 = [
                                float(x) for x in value.split(';')[1].split(',')]
                            # ic(fx, fy, cx, cy, k1, k2, p1, p2, k3)
                        if 'XMP:DewarpFlag' in tag and value != 0:
                            print(f"\t[*] {file} is already dewarped")
                            continue

            # Example from Phantom 4 RTK
            # Dewarp Data                     : 2018-09-04; 3678.870000000000, 3671.840000000000,
            #                                   10.100000000000, 27.290000000000, -0.268652000000,
            #                                   0.114663000000,0.000015268800,-0.000046070700,
            #                                  -0.035026100000
            # Dewarp Flag                     : 0
            # Date and Time, fx, fy, cx, cy, k1, k2, p1, p2, k3

            # dist_coeff[0, 0] = -0.268652000000    // k1
            # dist_coeff[1, 0] = 0.114663000000     // k2
            # dist_coeff[2, 0] = 0.000015268800     // p1
            # dist_coeff[3, 0] = -0.000046070700    // p2
            # dist_coeff[4, 0] = -0.035026100000    // k3
            # cam1 = np.zeros((3, 3), np.float32)
            # cam1[0, 2] = 2736 - 10.100000000000   // cX (5472x3648 Width / 2)
            # cam1[1, 2] = 1824 + 27.290000000000   // cY (5472x3648 Height / 2)
            # cam1[0, 0] = 3678.870000000000        // fx
            # cam1[1, 1] = 3671.840000000000        // fy

            dist_coeff = np.zeros((5, 1), np.float64)
            dist_coeff[0, 0] = k1
            dist_coeff[1, 0] = k2
            dist_coeff[2, 0] = p1
            dist_coeff[3, 0] = p2
            dist_coeff[4, 0] = k3
            cam1 = np.zeros((3, 3), np.float32)

            cam1[0, 2] = input_image.shape[1]/2 + cx
            cam1[1, 2] = input_image.shape[0]/2 + cy
            cam1[0, 0] = fx
            cam1[1, 1] = fy
            cam1[2, 2] = 1

            map1, map2 = cv2.initUndistortRectifyMap(
                cam1, dist_coeff, None, cam1, input_image.shape[1::-1], cv2.CV_32FC1)
            dewarped_image = cv2.remap(
                input_image, map1, map2, cv2.INTER_LINEAR)

            file_name_base, file_extension = os.path.splitext(
                os.path.split(file)[1])

            new_file_name = os.path.join(
                output_dir, f'{file_name_base}_dewarped{file_extension}')

            print(f"\t[*]  Write: {new_file_name}")
            cv2.imwrite(new_file_name, dewarped_image)

            print("\t[*] Copy  EXIF")
            with ExifTool() as et:
                et.execute('-TagsFromFile', file, '-all:all',
                           new_file_name, '-overwrite_original')


if __name__ == "__main__":
    main()
