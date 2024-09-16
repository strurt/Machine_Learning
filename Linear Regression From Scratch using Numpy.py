# tutorial taken from : "https://madewithml.com/courses/foundations/linear-regression/"

import numpy as np 
import pandas as pd 

## linear regression using numpy from scratch 
## Generating a random dataset 

N = 50 ; 
def generate_data(num_samples):
    x = np.array(range(num_samples))
    noise = np.random.uniform(-10, 20 , size = num_samples)
    y = 3.5 * x + noise 
    return x,  y 

# the function above returns a numpy array x in the range of 0 - 50 and numpy array y calculated by multiplying x with 3.5 plus adding a random noise

x, y = generate_data(N)
data = np.column_stack([x, y ])
print(data[:5])

# the code above stacks the two numpy arrays toegther

df = pd.DataFrame(data, columns = ['x','y'])
x= df[['x']].values
y = df[['y']].values
print(x)
print(y)
print ( df.head())

# the code above converts the stacked numpy arrays into a pandas dataframe

import matplotlib.pyplot as plt
plt.title("Scatterplot")
plt.scatter(x = df['x'], y = df['y'])
plt.show()

# the code above creates a scatterplot to visualise the points in the dataset

# we now divide the dataset into three parts : train, test and validation
# train to train the model, val to validate the model during training and test to evaluate the fully trained model 

train_size = 0.8 
test_size = 0.1
val_size = 0.1 

# we first shuffle the dataset before splitting 

index = list(range(N))
np.random.shuffle(index)
# numpy fancy indexing 
x = x[index]
y = y[index]

# now we split :  first define the range of train, test and val 

train_start = 0 
train_end = int(train_size * N )
val_start = train_end
val_end = int ( (train_size + val_size) * N )
test_start = val_end

# now split 

x_train = x[train_start: train_end]
x_val = x[val_start : val_end]
x_test = x[test_start : ]
y_train = y[train_start : train_end ]
y_val = y[val_start : val_end ]
y_test = y[test_start :]

# now lets look at the shape of the train, test and val sets 

print(x_train.shape, x_test.shape, x_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)

# now we standardise
# to standardise we need to make the mean equal to 0 and the standard deviation to 1 
# it is done to ensure all the features are on a similar scale, to prevent a large scale feature from dominating the small scale feature
# it is different from normalisation as normalisation constricts the features to a particular range value . mostly [0, 1 ]
# formula for standrdsisation is ( ignore the spelling :) ) 
# Xs = ( X - M ) / sd , where M is mean , Sd is standard deviation 
# sd = summation of ( each values - mean ) ^ 2 divided by the number of values
# formula for normalisation is : 
# Xs = ( X - Xmin / Xmax - Xmin )


# standardisation : 
def standardisation ( data , mean , std ): 
    return ( data - mean )/std 

x_mean = np.mean(x_train)
y_mean = np.mean(y_train)
x_sd = np.std(x_train)
y_sd = np.std(y_train)

# we compute mean and sd only for training set as val and tests are considered as unknown datasets during model training 
# now we standardise : 
x_train = standardisation(x_train, x_mean, x_sd)
x_val = standardisation(x_val, x_mean, x_sd)
x_test = standardisation(x_test, x_mean, x_sd)
y_train = standardisation(y_train, y_mean, y_sd)
y_val = standardisation(y_val, y_mean, y_sd)
y_test = standardisation(y_test, y_mean , y_sd)


# now we generate weight  : 
input_dimension = x_train.shape[1]
output_dimension = y_train.shape[1]
Weight = 0.01 * np.random.randn(input_dimension, output_dimension)
# shape : ( row , column ) where each row reperesnets a datapoint and each column represents a feature associated with the datapoint 
# shape ( 10 , 4) : means 10 row , 4 col. each row is a datapoint having 4 feature
Bias = np.zeros((1, 1))

# now we find the y_pred : ( y = mx + c : m is weight here , c is bias )
y_pred = np.dot(x_train , Weight ) + Bias 
print ( " the first y_pred calucated is ", y_pred )
print(y_pred.shape)

# now we find the loss function for the predicted y's 
# MSE = 1 / N * Summation ( y - ypred)^2
loss = ( 1 / len ( y_train)) *np.sum((y_train - y_pred) ** 2)
print(" the first loss calculated is " , loss )

# backpropagation / optimisation
# calulcate grdient loss -> 
#we find the derivative of loss wrt to weights and bias i.e we find out how to change Weight and Bias so as to reduce the loss function 
#gradient loss wrt weight : -(2/N)*summation(y-y_pred)x
#gradient loss wrt bias : -(2/N)*summation(y-y_pred)1

dw = -(2 / len(y_train)) * np.sum((y_train - y_pred) * x_train)
db = -(2 / len(y_train)) * np.sum((y_train - y_pred) * 1 )

# the formula to update weights is :  
# new_weight = weight - learning_rate(dw)
#new_bias = bias - learning_rate(db)
# learnind rate allows us to control our weight updation 
learning_rate = 1e-1 
Weight = Weight -(learning_rate*( dw))
Bias = Bias - (learning_rate*(db))
print(Weight)
print(Bias)

# now we repeate backpropagation for number of times to reduce the loss and train the model 
epoch = 500 
for i in range ( epoch):
    y_pred = np.dot(x_train , Weight ) + Bias 
    loss = ( 1 / len ( y_train)) *np.sum((y_train - y_pred) ** 2)
    if ( i % 5 == 0 ): 
        print( i , loss )
    dw = -(2 / len(y_train)) * np.sum((y_train - y_pred) * x_train)
    db = -(2 / len(y_train)) * np.sum((y_train - y_pred) * 1 )
    Weight = Weight -(learning_rate*( dw))
    Bias = Bias - (learning_rate*(db))

print( " the loss after training is " , loss)
print ( " y_pred after training is ", y_pred)

print ( Weight.shape)
print(Bias.shape)
print(x_train.shape)
print(x_test.shape)


# now we evaluate the model on the upadted weights from training : 
# We calcuate the 'y_pred' for both the training and the testing set and
# find the loss function for both the sets respectively 
y_pred_eval_train = np.dot(x_train, Weight) + Bias
y_pred_eval_test  = np.dot(x_test, Weight )+Bias
train_mse = np.mean((y_train - y_pred_eval_train)**2)
test_mse = np.mean((y_test - y_pred_eval_test ) **2 )
print ( train_mse, test_mse)

# we plot graph for train predictions :: 
plt.figure( figsize = ( 15, 5 ))
plt.title( " train_result ")
plt.scatter ( x_train , y_train , label = ' Train set' )
plt.plot(x_train , y_pred_eval_train, label = ' Train model')
plt.show()

# we plot graph for test predictions : 
plt.figure ( figsize=( 15, 5))
plt.title("Test Result ")
plt.scatter ( x_test, y_test, label = 'Test set')
plt.plot ( x_test, y_pred_eval_test, label = ' test model ')
plt.show( )















