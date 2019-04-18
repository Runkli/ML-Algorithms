

#basic regression line code


from statistics import mean
import numpy as npW
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xaxis = np.array([1,2,3,4,5,6]) 
yaxis = np.array([5,4,6,5,6,7]) #some random arrays

def best_fit_slope_and_intercept(xaxis,yaxis):
    m = ( ((mean(xaxis)*mean(yaxis))-mean(xaxis*yaxis))/
         ((mean(xaxis)*mean(xaxis))-mean(xaxis*xaxis)) )     #formula for finding the slope
    b = mean(yaxis)- m*mean(xaxis)             #formula to find the intercept
    return m,b           



m,b = best_fit_slope_and_intercept(xaxis,yaxis) #assign the outputs to some variables

print(m,b)

regression_line = [(m*x)+b for x in xaxis]    #finding the best fit with the xs that we have

predict_x = 8
predict_y = (m*predict_x)+b          #some random value to predict

plt.scatter(xaxis,yaxis)
plt.scatter(predict_x,predict_y,color='g')
plt.plot(xaxis,regression_line)
                   
plt.show()                    #visualising the outputs