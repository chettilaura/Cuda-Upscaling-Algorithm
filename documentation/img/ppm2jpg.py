# -*- coding: utf-8 -*-

import os
import sys
import time
from PIL import Image

def ppm2jpg(fp, out='./'):
    fbsname = os.path.basename(fp)
    fname, fext = os.path.splitext(fbsname)
    #print('%s, %s' % (fname, fext))
    if fext != '.ppm':
        raise Exception('Not ppm !')
    if not os.path.exists(out):
        os.makedirs(out)
    elif not os.path.isdir(out):
       raise Exception('Out path is not a directory !')
    fsave = os.path.join(out, '%s.jpg' % fname)
    #print(fsave)
    img = Image.open(fp)
    img.save(fsave)

def ppms2jpgs(dp, out='./'):
    #print('\t%s' % dp)
    if not os.path.isdir(dp):
        print('\tPath provided is not a directory')
        return
    fps = os.listdir(dp)
    for fp in fps:
        fp = os.path.join(dp, fp)
        print('\tConverting %s ...' % fp, end="")
        try:
            #time.sleep(0.1)
            ppm2jpg(fp, out)
            print('ok.')
        except Exception as E:
            print('error: %s' % E)

if __name__ == "__main__":
    try:
        dp = sys.argv[1]
    except:
        print('\tInput folder path is required !')
        exit()
    try:
        out = sys.argv[2]
    except:
        out = './'

    ppms2jpgs(dp, out)
