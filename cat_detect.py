"""This file applies the trained color detector to a given image"""
import argparse
import os
import glob
import re
import sys
import cv2 as cv
from PIL import ImageColor, Image


def parsearguments():
    parser = argparse.ArgumentParser(
        description='object detection using cascade classifier')
    parser.add_argument('-i', '--image', help='image file name')
    parser.add_argument('-c', '--cascade', dest='cascadefilename', help='cascade file name',
                        default='models/cat/lbp/cascade.xml')
    parser.add_argument('-s', '--scalefactor',
                        dest='scalefactor', type=float, default=1.1)
    parser.add_argument('-m', '--minneighbors',
                        dest='minneighbors', type=int, default=3)
    parser.add_argument('-o', '--output', dest='output',
                        default='box/detect.jpg')
    return parser.parse_args()


def detect(imagefilename, cascadefilename, scalefactor, minneighbors, categoly):
    if not os.path.isdir('faces/'+categoly):
        os.mkdir('faces/'+categoly)
    srcimg = cv.imread(imagefilename)
    root, ext = os.path.splitext(os.path.basename(imagefilename))
    #ssimg = Image.open(imagefilename)
    if srcimg is None:
        print('cannot load image '+imagefilename)
        sys.exit(-1)
    cascade = cv.CascadeClassifier(cascadefilename)
    objects = cascade.detectMultiScale(srcimg, scalefactor, minneighbors)
    count = len(objects)
    print('detection count: %s' % (count,))
    i = 0
    for (x, y, w, h) in objects:
        nx = int(x - round(w / 10))
        if (nx < 0):
            nx = 0
        ny = int(y - round(h / 10))
        if (ny < 0):
            ny = 0
        nw = int(w + 2*(round(w / 10)))
        nh = int(h + 2*(round(h / 10)))

        dst = srcimg[ny:(ny+nh), nx:(nx+nw)]
        facename = 'faces/' + categoly + '/' + root + '_' + str(i) + ext
        # print(facename)
        cv.imwrite(facename, dst)
        i += 1
    return srcimg


if __name__ == '__main__':
    args = parsearguments()
    print('cascade file: %s' % (args.cascadefilename,))
    imagedir = 'catimage'
    pattern = re.compile('.*[.](jpg|jpeg|png|bmp|gif)$')
  #  images = [image for image in os.listdir(imagedir) if re.match(pattern, image)]
    dir_list = os.listdir(imagedir)
    for categoly in (dir_list):
        if categoly == '.DS_Store':
            continue
        images = glob.glob(imagedir + "/" + categoly + "/*.jpg")
        print('target files: %s' % (len(images), ))
        for i, image in enumerate(images):
            imagesrc = images[i]
            result = detect(imagesrc, args.cascadefilename,
                            args.scalefactor, args.minneighbors, categoly)
