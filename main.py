#Predicting car selling prices
import numpy as np
import pandas as ps
import sklearn.metrics

dataset=ps.read_csv("carData.csv")
# print(dataset.shape)
print(dataset.info())
# print(dataset.isnull().sum())


#checking indivdual values

#print(dataset.Fuel_Type.value_counts())

#encoding each value with replace with numerical value


#print(dataset.Seller_Type.value_counts())

dataset.replace({"Seller_Type":{"Dealer":0,"Individual":1}},inplace=True)
dataset.replace({"Fuel_Type":{"Petrol":0,"Diesel":1,"CNG":2}},inplace=True)

#print(dataset.Transmission.value_counts())

dataset.replace({"Transmission":{"Manual":0,"Automatic":1}},inplace=True)


print(dataset.describe())
"""After replaced string with int split X and Y"""

X=dataset.drop(["Car_Name","Selling_Price"],axis=1)

Y=dataset["Selling_Price"]


print(dataset.groupby("Fuel_Type").mean())

#print("XX :",X.info())

"""Split dataset to training set to test set"""

from sklearn.model_selection import train_test_split

x_train ,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)

#print(Y)
"""Data preprocessing is over and we need to make Regression"""

from sklearn.linear_model import LinearRegression

linear=LinearRegression()

from sklearn.linear_model import Lasso
linear2=Lasso()
#print(x_train["Fuel_Types"])
linear.fit(x_train,y_train)
linear2.fit(x_test,y_test)
predicted_price_xtrain = linear.predict(x_train)
predicted_price_xtest=linear2.predict(x_test)

#rint(x_train[:,:-1])



from sklearn import metrics
"""R squared Error   Comparing both matrix"""
error_score=metrics.r2_score(y_train,predicted_price_xtrain)

print("Error is ",error_score)


"""We may find out error but not accuracy   --    raise ValueError("{0} is not supported".format(y_type))
ValueError: continuous is not supported"""
# from sklearn.metrics import accuracy_score
#
# score=accuracy_score(y_train,predicted_price_xtrain)
#
# print("Score accu :",score)
#


import matplotlib.pyplot as plt

# plt.scatter(y_train,predicted_price_xtrain,color="red")
# plt.plot(predicted_price_xtrain)
# plt.xlabel("actual price")
# plt.ylabel("predicted price")
# plt.show()
#
# plt.scatter(y_test,predicted_price_xtest,color="green")
# plt.xlabel("actual price")
# plt.ylabel("predicted price")
# plt.show()
#arr=[[y_train],[predicted_price_xtrain]]
#print(arr)


loc=np.asarray(predicted_price_xtrain,x_train)


print(loc)
print(x_train[1:5])
print(y_train[1:5])


#print(dataset.head())