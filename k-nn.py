from operator import itemgetter
from random import shuffle
import numpy as np
from sklearn.metrics import confusion_matrix
from math import *
from scipy.spatial import distance

def readData(file):
    rfile = open(file, "r")

    l= []
    for line in rfile:
        stripped_line = line.strip()
        item = stripped_line.split(';')
        l.append(item)

    rfile.close()
    return l

def formatData(lst):
    for item in lst:
        for i in range(len(item)-1):
            item[i] = float(item[i])
    return lst

def formatTestData(lst):
    for item in lst:
        for i in range(len(item)):
            item[i] = float(item[i])
    return lst

def norm(data_1, data_2, data_len):
    dist = 0
    for i in range(data_len):
        dist = dist + np.square(data_1[i] - data_2[i])
    return np.sqrt(dist)

def divideData(lst, prop):
    shuffle(lst)
    pop, test = lst[:int(prop*len(lst))], lst[int(prop*len(lst)):]
    pop_class, test_class = [],[]
    for i in range(len(pop)):
        pop_class.append(pop[i][-1::])
        pop[i] = pop[i]
        
    for i in range(len(test)):
        test_class.append(test[i][-1::])
        test[i] = test[i]
    return pop, test, pop_class, test_class

def knn(dataset, testInstance, k): 
    distances = [None] * len(dataset)
    length = len(testInstance)
    for x in range(len(dataset)): 
        distances[x] = [x, distance.euclidean(testInstance, dataset[x][:-1])]
    
    sort_distances = sorted(distances, key = itemgetter(1))
    #print(sort_distances)
    neighbors = []
    
    for x in range(k):
        neighbors.append(sort_distances[x])
    
    counts = {0 : 0, 1 : 0}
    
    for x in range(len(neighbors)):
        response = dataset[neighbors[x][0]][-1]
        if response in counts:
            counts[response] += 1
        else:
            counts[response] = 1
    
    sort_counts = {k: v for k, v in sorted(counts.items(), key=lambda item: item[1], reverse=True)}
    #print(sort_counts)
    return(list(sort_counts.keys())[0])

def confusionMatrix(actu, pred):
    return confusion_matrix(actu, pred)

def outputFile(data):
    file = open("prediction.txt", "w+")
    for i in range(len(data)):
        file.write(data[i] + "\n")
        
    file.close()
    

if __name__ == '__main__' :
    dataset = formatData(readData("C:/Users/33769/Documents/Python Scripts/KNN/data.txt"))
    preTest = formatData(readData("C:/Users/33769/Documents/Python Scripts/KNN/preTest.txt"))
    finalTest = formatTestData(readData("C:/Users/33769/Documents/Python Scripts/KNN/finalTest.txt"))
    
    
        

    #training
    
    #dev, _, dev_class, _ = divideData(preTest, 1)
    # k_n = [1, 3, 5, 9, 11, 13, 15, 17, 19, 21, 25, 27, 29, 31, 33, 35]
    # dev_set_k = {}
    
    # for k in k_n:
    #     dev_set = [None] * len(dev)
    #     for i in range(len(dev)):
    #         dev_set[i] = knn(dataset, dev[i][0:10], k)
            
    #     dev_set_k[k] = dev_set
    #     #print(dev_set)
    
    # for k in k_n:
    #     count = 0
    #     for i in range(len(dev)):
    #         if(dev_set_k[k][i] == dev_class[i][0]):
    #             count += 1
        
    #     print(k,' : ',count/len(dev))
        
    #Conclusion : best k = 21
    
    ###############################################
    
    #testing
    
    # k = 21
    # # test_set = [None] * len(test)
    # # for i in range(len(test)):
    # #     test_set[i] = knn(dataset, test[i][0:10], k)
    
    # # count = 0
    # # for i,j in zip(test_class, test_set):
    # #     if i[0] == j:
    # #         count += 1
    # #     else:
    # #         pass
        
    # # print('Final Accuracy of the Test dataset is ', count/(len(test_class)))
    
    # # print(confusionMatrix([item[0] for item in test_class], test_set))
    
    ###############################################
    
    #predicting
    
    k = 21
    results = [None] * len(finalTest)
    for i in range(len(finalTest)):
        results[i] = knn(dataset, finalTest[i], k)
    outputFile(results)
    