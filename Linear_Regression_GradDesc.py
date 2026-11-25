'''
Linear Regression Via Gradient Descent 

'''
import numpy as np 
import matplotlib.pyplot as plt 


## plotting functions
def Graph(x,y,m,b):
    plt.plot(x,m*x + b,'r')
    plt.scatter(x,y)
    plt.grid(True)
    plt.title(" Linear Regression ")
    plt.xlabel(" X points ")
    plt.ylabel(" Data Points ")
    plt.show()

def LinReg(x,y,learning_rate = 0.01,epochs= 1000):
    # y = mx + b form
    m = 0
    b = 0
    n = len(x)
    loss_list =[]
    ## Grad Desc
    for i in range(epochs):
        y_pred = m*x + b
        err = (y_pred - y ) 
        loss = np.sum((err)**2)
        loss_list.append(loss)
        if i > 2:
            # Exit condition when it goes up 
            if loss_list[i-1] < loss_list[i]:
                Graph(x,y,m,b)
                ## Return Values 
                return m,b
                
        dm = (2/n)*np.dot(err,x)
        db =(2/n)*np.sum(err)
        m = m - dm*learning_rate
        b = b - db*learning_rate
     ## if it didnt went up ie after exhaustion of epochs  
    print(f"The minima wasn't reached in {epochs} epochs; returning last values.")
    Graph(x,y,m,b)
    return m, b

def Test():
    x = np.arange(2,5,0.1)
    y = np.log(x)
    print(LinReg(x,y,learning_rate = 0.01,epochs= 1000))
Test()