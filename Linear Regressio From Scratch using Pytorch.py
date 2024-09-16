import numpy as np 
import pandas as pd 

# generate a random numpy array  : 
N = 100 
def random_dataset(no_of_samples):
    x = np.array(range(no_of_samples))
    noise = np.random.uniform(-10, 20 , no_of_samples)
    y = 3.5  * x + noise 
    return x, y 

x, y = random_dataset(N)
data = np.column_stack([x, y])
print ( data[:5])

# convert the stacked numpy array into a pandas dataset and convert the cols into a numpy array : 
df = pd.DataFrame(data, columns= ['x', 'y'])
x = df[['x']].values 
y = df[['y']].values
print(df.head(10)) 
print(x.shape)
print(y.shape)
print(x)
print(y)

#plot the dataset 
import matplotlib.pyplot as plt 
plt.title("Scatterplot")
plt.scatter(x , y)
plt.show()


# now divide the dataset into training, testing and validation 
# do this using the sklearn train test split 
# we just need the train and test here but doing it anyway 
import sklearn
from sklearn.model_selection import train_test_split
x_train, x_temp, y_train , y_temp = train_test_split(x , y , train_size=0.8)
val_size = 0.1
test_size = 0.1 
val_test_split = val_size / ( val_size + test_size )
x_val , x_test, y_val , y_test = train_test_split(x_temp, y_temp , train_size=val_test_split)


# now we standardise the dataset 
# do this using the standard scaler library 

from sklearn.preprocessing import StandardScaler

#generate x_scaler , y_scaler ( basically computing the sd and mean and standardising )
x_scaler = StandardScaler().fit(x_train)
y_scaler = StandardScaler().fit(y_train)

# fit the scaler onto your dataset : 
x_train = x_scaler.transform(x_train)
y_train = y_scaler.transform(y_train)
x_test = x_scaler.transform(x_test)
y_test = y_scaler.transform(y_test)
x_val = x_scaler.transform(x_val)
y_val = y_scaler.transform(y_val)

# convert the numpy arrays into tensors : 
import torch 
x_train = torch.tensor(x_train, dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.float32)
x_test = torch.tensor(x_test, dtype = torch.float32)
y_test = torch.tensor(y_test, dtype = torch.float32)
x_val = torch.tensor(x_val, dtype = torch.float32)
y_val = torch.tensor(y_val, dtype = torch.float32)

# now using the nn libary to initialise weights and build your linear model 

from torch import nn 
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]
print ( input_dim , output_dim)
model = nn.Linear(input_dim, output_dim)

# now fit the model into your trainig dataset 
y_pred = model(x_train)
print(y_pred)

#now calcualte your loss 
loss_function = nn.MSELoss()
loss = loss_function(y_pred, y_train)
print(loss)

# now for backpropagation use an optimiser : 
from torch.optim import Adam
#set a learning rate 
learning_rate = 1e-1
Optimiser = Adam(model.parameters(), lr = learning_rate)

# now model is trained over several iterations : 

epoch = 500 
for i in range ( epoch ):
    y_pred  = model( x_train)
    loss = loss_function(y_pred, y_train)

    Optimiser.zero_grad()
    loss.backward()
    Optimiser.step()
    if ( i % 5 == 0 ):
        print( loss, i )

# model evaluation : 
y_pred_train = model( x_train)
y_pred_test = model(x_test)
#find train and test loss 
loss_y_pred_train = loss_function(y_pred_train , y_train )
loss_y_pred_test = loss_function(y_pred_test, y_test)
print ( loss_y_pred_train , loss_y_pred_test)

# now both the trainig and eval is graphed 
# pytoch tensors cannot be directly graphed, so you need to convert them back to numpy arrays

x_train = x_train.detach().numpy()
x_test = x_test.detach().numpy()
y_train = y_train.detach().numpy()
y_test = y_test.detach().numpy()
y_pred_train = y_pred_train.detach().numpy()
y_pred_test = y_pred_test.detach().numpy()


#training set 
plt.figure ( figsize=(15, 5))
plt.title("train")
plt.scatter ( x_train , y_train , label = 'train set')
plt.plot( x_train , y_pred_train , label = 'train model ')
plt.show ( )

#testing set 
plt.figure ( figsize = ( 15, 5))
plt.title ( " test set ")
plt.scatter ( x_test, y_test, label = 'train set ')
plt.plot ( x_test , y_pred_test, label = "train model ")
plt.show( )



# basically in pytorch : 
# train_test_split takes care of your splits 
# standardscaler takes care of your scaling
# nn library takes care of your weights
# torch.optim takes care of your bakpropagation 
# nn.MSELoss takes care of your MSE loss functions 
  








