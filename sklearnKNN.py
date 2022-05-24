from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

def readData(file):
    rfile = open(file, "r")

    l= []
    for line in rfile:
        stripped_line = line.strip()
        item = stripped_line.split(';')
        l.append(item)

    rfile.close()
    return l

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

def formatData(lst):
    for item in lst:
        for i in range(len(item)-1):
            item[i] = float(item[i])
    return lst

dataset = formatData(readData("C:/Users/33769/Documents/Python Scripts/KNN/data.txt"))
preTest = formatData(readData("C:/Users/33769/Documents/Python Scripts/KNN/preTest.txt"))

X_train = [i[0:10] for i in dataset]
y_train = [i[10] for i in dataset]
X_test = [i[0:10] for i in preTest]
y_test = [i[10] for i in preTest]
 
neighbors = np.arange(1, 50)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
 
# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
     
    # Compute training and test data accuracy
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

print(train_accuracy)
print(test_accuracy)

# Generate plot
plt.plot(neighbors, test_accuracy, label = 'preTest dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'data dataset Accuracy')
 
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
    
    