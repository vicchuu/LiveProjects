#Predicting car selling prices


import pandas as ps


dataset=ps.read_csv("carData.csv")
# print(dataset.shape)
print(dataset.info())
# print(dataset.isnull().sum())


#checking indivdual values

#print(dataset.Fuel_Type.value_counts())

#encoding each value with replace with numerical value

dataset.replace({"Fuel_Types":{"Petrol":0,"Diesel":1,"CNG":2}},inplace=True)


#print(dataset.Seller_Type.value_counts())

dataset.replace({"Seller_Type":{"Dealer":0,"Individual":1}},inplace=True)


#print(dataset.Transmission.value_counts())

dataset.replace({"Transmission":{"Manual":0,"Automatic":1}},inplace=True)


print(dataset.head())