import numpy as np
def ID1():
    '''
        Write your personal ID here.
    '''
    # Insert your ID here
    return 319095832
def ID2():
    '''
        Only If you were allowed to work in a pair will you fill this section and place the personal id of your partner otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem
    X*theta=y using the least squares method
    :param X: input matrix
    :param y: input vector
    :return: theta = (Xt*X)^(-1) * Xt * y 
  '''
  X_t = np.transpose(X)
  theta = np.linalg.inv(X_t @ X) @ X_t @ y
  return theta
  

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: input matrix
    :param s: input ground truth label
    :return: accuracy of the model
  '''
  pred_s = model.predict(X)
  correct_pred = (pred_s == s).sum()
  total_pred = len(s)
  accuracy = (correct_pred/total_pred)*100
  return accuracy

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem.  
  '''
  return [-0.0598359,   0.03645439, -0.02617194,  0.01224763, -0.03096861, -0.03254863,
  0.09071795,  0.01229223, -0.009611,   -0.03616204,  0.06700759,  0.01730464,
  0.08771157,  0.14135295,  0.76807147,  0.04201202,  0.00840565,  0.01225948,
  0.00498126,  0.00541578,  0.0321941,   0.01593336, -0.00443764, -0.02644167,
 -0.02392134,  0.02088,    -0.01358706, -0.0173445]

def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return -2.2411126107882224e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the classification problem.  
  '''
  return [[-0.14590677, -0.10898822,  0.19421773, -0.10698311, -0.3046274,  -0.49625283,
  -0.05159349,  0.01921418,  0.0054042,   0.27885861, -0.58830057, -0.02237805,
  -0.15471841,  1.05340682,  3.11899748, -0.43857465, -0.01557295,  0.03152631,
  -0.34302173, -0.16716791,  0.23312679, -0.00334467,  0.02359932, -0.16316622,
  -0.34669699,  0.02465231,  0.15658221,  0.40725705]]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return [0.22606728]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem.  
  '''
  return [0, 1]