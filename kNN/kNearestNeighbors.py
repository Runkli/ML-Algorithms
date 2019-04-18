from scipy.spatial import distance

#distance calculator function
def euc(a,b):
    return distance.euclidean(a,b)


#kNN classifier

class testKNN():
    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
        pass
    
    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    def closest(self,row):  #goes through rows and checks which row is the closest to the inputs
        best_dist = euc(row,self.x_train[0])
        best_index = 0
        for i in range (1, len(self.x_train)):   
            dist = euc(row, self.x_train[i])
            if dist<best_dist:
                best_dist = dist
                best_index = i
            else:
                pass
        return y_train[best_index]
        
    


from sklearn import datasets
iris = datasets.load_iris()   #sample test on the iris dataset

x = iris.data      #initialize x and y as the inputs and outputs for ease of use
y = iris.target

from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .5) #split given data in half, for training and testing

from sklearn.neighbors import KNeighborsClassifier
my_clf = testKNN()

my_clf.fit(x_train, y_train)   #using fit method to train the algorithm

predictions = my_clf.predict(x_test) #predicting

from sklearn.metrics import accuracy_score

print (accuracy_score(y_test,predictions))   #calculating the accuracy score 
