import os
import numpy as np

def writeData(filepath, contentToCopy):
    dataPartition = open(filepath, "w+")
    for element in contentToCopy:
        dataPartition.write(element)


if __name__ == '__main__':
    
    trainPerc = 0.5
    testPerc = 0.25

    documents = []

    with open("../Dataset/dataset.lst", "r") as datafile:
        line = datafile.readline()
        while line:
            documents.append(line)
            line = datafile.readline()

    numofDocs = len(documents)
    np.random.shuffle(documents)

    trainSlice = documents[:int(numofDocs*trainPerc)]
    valSlice = documents[int(numofDocs*trainPerc):int(numofDocs*trainPerc) + int(numofDocs*testPerc)]
    testSlice = documents[int(numofDocs*trainPerc)+int(numofDocs*testPerc):]
    writeData("train.lst", trainSlice)
    writeData("validation.lst", valSlice)
    writeData("test.lst", testSlice)

    
        
    
    


