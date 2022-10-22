# importing necessary libraries
import pickle
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from skimage.transform import resize
import cv2

# preprocesing and general function implementations 

#defining base directory
directory_base_address = 'D:\TAU\Courses\Intro to ML\Exercise week 4\cifar-10-batches-py'

# file reading function
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

# declare empty lists to store training data from different files
X =[]
Y =[]

# reading training data
for i in range(1,6):
    datadict = unpickle(directory_base_address+'/data_batch_'+str(i))
    X.append(datadict["data"])
    Y.append(datadict["labels"])

# reading test data
datadict_test = unpickle(directory_base_address+'/test_batch')

# merging all the training data into one    
X_train = np.concatenate(X, axis=0 )
Y_train = Y[0]+Y[1]+Y[2]+Y[3]+Y[4]

X_test= datadict_test["data"]
Y_test = datadict_test["labels"]

labeldict = unpickle(directory_base_address+'/batches.meta')
label_names = labeldict["label_names"]

X_train = X_train.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y_train = np.array(Y_train)

X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y_test = np.array(Y_test)


#CIFAR-10:  Evaluation Function
def class_acc(pred,gt):
    
    correctly_classified = 0
    total_sample = len(pred)
    
    for i in range(total_sample):
        if pred[i] == gt[i]:
            correctly_classified=correctly_classified+1
            
    accuracy = (correctly_classified/total_sample)*100
    
    return accuracy



#-----------------Task 1-----------------------------
    
# Resizing function 1*1
def cifar10_color(X):
    
    Xp = np.zeros((X.shape[0],3))
    
    for i in range(X.shape[0]):
       image = X[i]  
       image_resize = resize(image,(1,1))
       Xp[i] = image_resize
    
    return Xp


Xp_train = cifar10_color(X_train)
Xp_test = cifar10_color(X_test)


# Function to compute(learn) the normal distribution parameters.    
def cifar10_naivebayes_learn(Xp,Y):
    means =[]
    covariances =[]
    priors =[]
    
    for classNumber in range(0,10):
        filter_Y = Y == classNumber;
        
        filtered_Xp = Xp[filter_Y];
        
        prior=((filtered_Xp.shape[0])/(Xp.shape[0]));
        getMean = np.mean(filtered_Xp,axis=0)
        getVar = np.var(filtered_Xp, axis=0)
        
        means.append(getMean)
        covariances.append(getVar)
        priors.append(prior)
        
    return np.array(means),np.array(covariances),np.array(priors)
       

meanMatrix,varMatrix,priorMatrix = cifar10_naivebayes_learn(Xp_train,Y_train)

# Naive Bayes Classifier Function
def cifar10_classifier_naivebayes(x,mu,sigma,p):
    labels_naivebayes = []
    for i in range(x.shape[0]):
        probabilities = []
        temp_prob = norm.logpdf(x[i], mu,sigma)
        
        for j in range(10):
            probabilities.append(np.prod(temp_prob[j])*p[j])
        
        maximum_probabilities = max(probabilities)
        label = probabilities.index(maximum_probabilities)
        labels_naivebayes.append(label)
            
        
    accuracy = class_acc(np.array(labels_naivebayes), Y_test)
    
    return labels_naivebayes, accuracy       
    
labels_naivebayes, accuracy_naivebayes = cifar10_classifier_naivebayes(Xp_test,meanMatrix,varMatrix,priorMatrix)    
   
print("Naive Bayes Accuracy: ", round(accuracy_naivebayes, 2)) 

#------------------- Task 2 -----------------------------------

# Function to compute(learn) the normal distribution parameters.    
def cifar10_bayes_learn(Xf,Y):
    
    means =[]
    covariances =[]
    priors =[]
    
    for classNumber in range(0,10):
        filter_Y = Y == classNumber;
        
        filtered_Xf = Xf[filter_Y];
        
        prior=((filtered_Xf.shape[0])/(Xf.shape[0]));
        getMean = np.mean(filtered_Xf,axis=0)
        getVar = np.cov(filtered_Xf,rowvar=False )
        
        means.append(getMean)
        covariances.append(getVar)
        priors.append(prior)
        
    
    return np.array(means),np.array(covariances),np.array(priors)    

meanMatrixT2,varMatrixT2,priorMatrixT2 = cifar10_bayes_learn(Xp_train,Y_train)

# Bayesian Classifier Function
def cifar10_classifier_bayes(x,mu,sigma,p):
    labels_bayes = []
    
    for i in range(x.shape[0]):
        probabilities = []
        for j in range(10):
            temp_prob = multivariate_normal.pdf(x[i], mu[j,:],sigma[j,:,:])
            probabilities.append(temp_prob*p[j])
        
        maximum_probabilities = max(probabilities)
        label = probabilities.index(maximum_probabilities)
        labels_bayes.append(label)
            
    
    accuracy = class_acc(np.array(labels_bayes), Y_test)
    
    return labels_bayes, accuracy 

labels_bayes, accuracy_bayes = cifar10_classifier_bayes(Xp_test,meanMatrixT2,varMatrixT2,priorMatrixT2)

print("Bayes Accuracy: ", round(accuracy_bayes, 2))


# ------------------- Task 3 ----------------------------

#extended Resize Function for n*n with default size (2,2)

def cifar10_nXn_color(images, size=(2, 2)):

    resized_images = np.zeros([images.shape[0], size[0], size[1], 3])

    for i in range(images.shape[0]):
        resized_images[i, :, :, :] = cv2.resize(images[i, :, :, :], size)
    
    resized_images = np.reshape(resized_images, (resized_images.shape[0], -1))
    
    return np.array(resized_images, dtype='uint8')


# list of different image sizes
image_sizes = [1, 2, 4]
accuracies = []


for i in range(len(image_sizes)):
    Xp_train = cifar10_nXn_color(X_train, size= (image_sizes[i], image_sizes[i]))
    Xp_test = cifar10_nXn_color(X_test, size= (image_sizes[i], image_sizes[i]))
    
    meanMatrix, varMatrix, priorMatrix = cifar10_bayes_learn(Xp_train, Y_train)
    
    labels_bayes, class_accuracy = cifar10_classifier_bayes(Xp_test,meanMatrix,varMatrix,priorMatrix)
    
    accuracies.append(round(class_accuracy, 2))
    print(f"Accuracy for size {image_sizes[i]} * {image_sizes[i]} is: {class_accuracy}")


image_shapes = ['1x1',' 2x2', '4x4']

# Plotting the Graph
plt.plot(image_shapes, accuracies)
plt.title("Classification accuracy for different image size")
plt.xlabel("Image Size")
plt.ylabel("Accuracy %")
plt.show()  