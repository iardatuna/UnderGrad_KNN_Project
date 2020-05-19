#Ismail Arda Tuna
#240201031
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import wittgenstein as lw
import time
from sklearn.metrics import roc_auc_score

pd.set_option("chained_assignment",None)
data_frame = pd.read_csv("heart_data.csv")
#Part a


data = data_frame[["age","cp","trestbps","thalach","chol","target"]]

#Part b
  
data.age= data.age.apply(lambda x: "younger person" if x<= 55 else "older person")

#Part c

label_encoder = preprocessing.LabelEncoder()
categorical_columns = data.select_dtypes(['object']).columns
for i in range(len(categorical_columns)):
    data[categorical_columns[i]] = label_encoder.fit_transform(data[categorical_columns[i]])
print(data[categorical_columns])

#Part d

dataset = data[data.columns[0:5]]
target = data["target"]
ripper_dataset = data
x_train,x_test, y_train, y_test  = train_test_split(dataset,target, test_size=0.2, random_state = 123)
ripper_train,ripper_test =  train_test_split(ripper_dataset, test_size=0.2, random_state = 123)
print("")
print ("Train size for Decision Tree" + " ------------>  " + str(x_train.shape))
print("")
print ("Test size for Decision Tree" + "  ------------>  " + str(x_test.shape))
print("")
print("")
print ("Train size for Ripper" + " ------------>  " + str(ripper_train.shape))
print("")
print ("Test size for Ripper" + "  ------------>  " + str(ripper_test.shape))
print("")
#Part e

ripper_clf = lw.RIPPER() 
ripper_start_time = time.time()
ripper_clf.fit(ripper_train,class_feat="target",random_state = 123)
ripper_predict = ripper_clf.predict(ripper_test)
ripper_run_time = time.time()-ripper_start_time

#Part f

print("Decision Tree with Entropy")
print("")
entropy_data = DecisionTreeClassifier(criterion = "entropy",random_state = 123, max_depth=5) 
entropy_start_time = time.time()
entropy_data = entropy_data.fit(x_train, y_train)
entropy_pred_data = entropy_data.predict(x_test)
entropy_run_time = time.time()-entropy_start_time
print("Accuracy with Entropy: ",accuracy_score(y_test, entropy_pred_data))
print("")

#Part g

auc_ripper = roc_auc_score(ripper_test[ripper_test.columns[5]],ripper_predict)

auc_entropy = roc_auc_score(y_test,entropy_pred_data)

print("AUC value for Ripper"+ " ------------>  " + str(auc_ripper))
print("")
print("Run time for Ripper"+ " ------------>  " + str(format(ripper_run_time*1000,".4f"))+" ms")
print("")
print("AUC value for Entropy"+ " ------------>  " + str(auc_entropy))
print("")
print("Run time for Entropy"+ " ------------>  " + str(format(entropy_run_time*1000,".4f"))+" ms")









