#Ismail Arda Tuna
#240201031
import pandas as pd
import matplotlib.pyplot as plt
import math  
data_frame = pd.read_csv("covid.csv")

#Part a

def MinMaxNormalizationDataFrame(data_frame):
    data_frame.iloc[:,0] = data_frame.iloc[:,0].apply(lambda x: x / 5)
    data_frame.iloc[:,1] = data_frame.iloc[:,1].apply(lambda x: (x - 35.0) / (39.9 - 35.0)) 
    return data_frame

#Part b 

def KNN(train_data,test_data,distance_type,k):
    if(distance_type == 0): #Euclidean Distance Type
        euclidean_dist_list = []
        euclidean_value = 0
        zero_count=0
        one_count=0
        first_k_elements = []
        for i in range(len(train_data)):
            euclidean_value = Euclidean_Distance(train_data.iloc[i][0],train_data.iloc[i][1], test_data[0],test_data[1])
            euclidean_dist_list.append((tuple((euclidean_value,i))))
            euclidean_value = 0
        euclidean_dist_list.sort()
        for j in range(k):
            first_k_elements.append(euclidean_dist_list[j])
        for index in range(len(first_k_elements)):
            if (train_data.iloc[first_k_elements[index][1]][2]==0):
                zero_count +=1
            elif (train_data.iloc[first_k_elements[index][1]][2]==1):
                one_count +=1
        if (one_count > zero_count):
            predicted_value = 1
        else:
            predicted_value = 0
        return predicted_value    
    elif(distance_type == 1): #Manhattan Distance Type
        manhattan_dist_list = []
        manhattan_value = 0
        zero_count=0
        one_count=0
        first_k_elements = []
        for i in range(len(train_data)):
            manhattan_value = Manhattan_Distance(train_data.iloc[i][0],train_data.iloc[i][1], test_data[0],test_data[1])
            manhattan_dist_list.append(tuple((manhattan_value,i)))
            manhattan_value = 0
        manhattan_dist_list.sort()
        for j in range(k):
            first_k_elements.append(manhattan_dist_list[j])
        for index in range(len(first_k_elements)):
            if(train_data.iloc[first_k_elements[index][1]][2]==0):
                zero_count +=1
            elif(train_data.iloc[first_k_elements[index][1]][2]==1):
                one_count +=1
        if (one_count > zero_count):
            predicted_value = 1
        else:
            predicted_value = 0
        return predicted_value

#Part c

def Euclidean_Distance(x1,y1,x2,y2):
    euclidean_distance = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return euclidean_distance    

#Part d
    
def Manhattan_Distance(x1,y1,x2,y2):
    manhattan_distance = abs((x1-x2))+abs(y1-y2) 
    return manhattan_distance    

#Part e
    
def AccuracyOfTestSet(test_set,distance_type,k):
    predict_count=0
    for i in range(len(test_set)):
        if(KNN(data_frame, test_set[i], distance_type, k) == test_set[i][2]):
            predict_count+=1
    accuracy = (predict_count/len(test_set))*100
    return accuracy

#Part f
    
def AccuracyListForPlot(data_frame,test_set,distance_type):
    accuracy_list=[]
    for i in range(len(data_frame)):
        if i>0:
            accuracy_list.append(AccuracyOfTestSet(test_set, distance_type,i))
        else:
            continue
    return accuracy_list

test_set = [[5,39.0,1], [4,35.0,0], [3,38.0,0], [2,39.0,1], [1,35.0,0], [0,36.2,0], [5,39.0,1], [2,35.0,0], [3,38.9,1], [0,35.6,0]]
test_set_2 = [[5, 39.0, 1], [4, 35.0, 0], [3, 38.0, 0],
              [2, 39.0, 1], [1, 35.0, 0], [0, 36.2, 0],
              [5, 39.0, 1], [2, 35.0, 0], [3, 38.9, 1],
              [0, 35.6, 0], [4, 37.0, 0], [4, 36.0, 1],
              [3, 36.6, 0], [3, 36.6, 1], [4, 36.6, 1]]
test_set = pd.DataFrame(test_set)
test_set = MinMaxNormalizationDataFrame(test_set)
test_set = test_set.values.tolist()
test_set_2 = pd.DataFrame(test_set_2)
test_set_2 = MinMaxNormalizationDataFrame(test_set_2)
test_set_2 = test_set_2.values.tolist()
data_frame = MinMaxNormalizationDataFrame(data_frame)
#print(data_frame)

print("")
print("Accuracy for test_set with Euclidean Distance -> %"+str(AccuracyOfTestSet(test_set, 0, 3)))
print("")
print("Accuracy for test_set_2 with Euclidean Distance -> %"+str(AccuracyOfTestSet(test_set_2, 0, 3)))
print("")
print("Accuracy for test_set with Manhattan Distance -> %"+str(AccuracyOfTestSet(test_set, 1, 5)))
print("")
print("Accuracy for test_set_2 with Manhattan Distance -> %"+str(AccuracyOfTestSet(test_set_2, 1, 5)))
test_set_euclidean_accuracy_list = AccuracyListForPlot(data_frame,test_set, 0)
test_set_manhattan_accuracy_list = AccuracyListForPlot(data_frame,test_set, 1)
test_set_2_euclidean_accuracy_list = AccuracyListForPlot(data_frame,test_set_2, 0)
test_set_2_manhattan_accuracy_list = AccuracyListForPlot(data_frame,test_set_2, 1)

plt.xlabel("k-values")
plt.ylabel("Accuracy")
plt.title("Test Set's Accuracy with Euclidean Distance")
plt.plot(test_set_euclidean_accuracy_list)
plt.savefig('test_set_euclidean_accuracy_list.png')
plt.show()


plt.xlabel("k-values")
plt.ylabel("Accuracy")
plt.title("Test Set's Accuracy with Manhattan Distance")
plt.plot(test_set_manhattan_accuracy_list)
plt.savefig('test_set_manhattan_accuracy_list.png')
plt.show()

plt.xlabel("k-values")
plt.ylabel("Accuracy")
plt.title("Test Set 2's Accuracy with Euclidean Distance")
plt.plot(test_set_2_euclidean_accuracy_list)
plt.savefig('test_set_2_euclidean_accuracy_list.png')
plt.show()

plt.xlabel("k-values")
plt.ylabel("Accuracy")
plt.title("Test Set 2's Accuracy with Manhattan Distance")
plt.plot(test_set_2_manhattan_accuracy_list)
plt.savefig('test_set_2_manhattan_accuracy_list.png')
plt.show()

#Comments

#       According to Test Set with Euclidean Distance, all k-values have the best configuration for KNN 
# as shown in the graph because the accuracy levels are the same. Accuracy is %90 for all values. 

#       According to Test Set with Manhattan Distance, except k is 33. In other words, there is a small 
# deviation while k-value is 33. Rest of the k-values have the best configuration
# as shown in the graph because the accuracy levels are the same. Accuracy is %90 for the rest of all values. 

#       According to Test Set 2 with Euclidean Distance, while k-values in range [13,21] and [27,33] 
# KNN has the best configuration. For the given k-values the accuracy levels are the same.
# Accuracy is %80 for these values.
#       According to Test Set 2 with Manhattan Distance  while k-values in range [13,17] and [29,32],
# k is 1 and 27 KNN has the best configuration. For the given k-values the accuracy levels are the same.
# Accuracy is %80 for these values.

