import os
from os import path
import shutil
import sys


#### CONSTS ####
COPY_DIR = "Data"
FROM_DIR = "Corpus"


def copyImageAndData(pathToFile, fileName, ispng):
    imagename_no_extension = fileName.split('.')[0]
    if not ispng:
        imagename_no_extension = '_'.join(imagename_no_extension.split('_')[:-1])
    
    agnosticName = imagename_no_extension + ".agnostic"
    kernName = imagename_no_extension + ".krn"
    skmName = imagename_no_extension + ".skm"


    if path.isfile(pathToFile + fileName) != True or path.isfile(pathToFile + agnosticName) != True or path.isfile(pathToFile + kernName) != True or path.isfile(pathToFile + skmName) != True:
       return

    #shutil.copy(pathToFile + "/" + fileName, COPY_DIR) #Copy the image
    #shutil.copy(pathToFile + "/" + agnosticName, COPY_DIR) #Copy the agnostic file
    #shutil.copy(pathToFile + "/" + kernName, COPY_DIR) #Copy the kern file
    #shutil.copy(pathToFile + "/" + skmName, COPY_DIR) #Copy the skm file

    #Add this data to the list
    dataList.write("%s\t%s\t%s\t%s\n" % (pathToFile + fileName, pathToFile  + agnosticName, pathToFile + kernName, pathToFile  + skmName))


dataList = open("dataset.lst", "w+")

if __name__ == '__main__':

   print("Copying files to path " + COPY_DIR)

   for file in os.listdir("./Dataset/Data/"):
      if file.endswith(".png"):
         if file[0] != ".":
            copyImageAndData("./Dataset/Data/", file, True)
      elif file.endswith(".jpg"):
         if file[0] != ".":
            copyImageAndData("./Dataset/Data/", file, False)
   
   print("Finished copying files")
