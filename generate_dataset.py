#!/usr/bin/env python

import sys
import os.path
import csv
import shutil

# This is a tiny script to help you creating a CSV file from a face
# database with a similar hierarchie:
#
#  philipp@mango:~/facerec/data/at$ tree
#  .
#  |-- README
#  |-- s1
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  |-- s2
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#  ...
#  |-- s40
#  |   |-- 1.pgm
#  |   |-- ...
#  |   |-- 10.pgm
#

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print "usage: create_csv <base_path>"
        sys.exit(1)

    BASE_PATH=sys.argv[1]
    SEPARATOR=";"

    label = 1
    i=0
    f= open('facerec.txt','w')
    for dirname, dirnames, filenames in os.walk(BASE_PATH):
        for subdirname in dirnames:
            if subdirname[-6:] == "_train":
              continue
            subject_path = os.path.join(dirname, subdirname)
            sp=subject_path
            for filename in os.listdir(subject_path):
              if filename != ".DS_Store":
                dstroot=subdirname + "_train"
                train_path =os.path.join(dirname, dstroot)
                if not os.path.exists(train_path):
                  os.makedirs(train_path)
                abs_path = "%s/%s" % (subject_path, filename)
                shutil.copy(abs_path,train_path)
                train_path1 = "%s/%s" % (train_path, filename)
                f.write(str(label) + " " +subdirname + " " +train_path1)
                f.write("\n") 
            label = label + 1
f.close()