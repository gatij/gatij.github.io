---
layout:     post
title:      "Linear Regression Model in Python:from scratch"
subtitle:   "Linear Regression Model:Simple mathematical fun"
date:       2018-07-05 14:23:00
author:     "Gatij Jain"
header-img: "img/post-LR-26-7-18.gif"
header-mask: 0.3
catalog:    true
tags:
    - Regression
    - Python
    - Scikit
    - ML Model
---


**Linear Regression** is one of the easiest algorithms in machine learning. In this post we will explore this algorithm and we will implement it using Python from scratch.

As the name suggests this algorithm is applicable for Regression problems. Linear Regression is a **Linear Model**. Which means, we will establish a linear relationship between the input variables**(X)** and single output variable**(Y)**. When the input**(X)** is a single variable this model is called **Simple Linear Regression** and when there are mutiple input variables**(X)**, it is called **Multiple Linear Regression.**

## **Simple Linear Regression**

We discussed that Linear Regression is a simple model. Simple Linear Regression is the simplest model in machine learning.

## Model Representation

In this problem we have an input variable **- X** and one output variable **- Y**. And we want to build linear relationship between these variables. Here the input variable is called **Independent Variable** and the output variable is called **Dependent Variable**. We can define this linear relationship as follows:

  $$Y = \beta_0 + \beta_1X$$
 
The β1 is called a scale factor or **coefficient** and β0 is called **bias coefficient**. The bias coeffient gives an extra degree of freedom to this model. This equation is similar to the line equation y=mx+b with m=β1(Slope) and b=β0(Intercept). So in this Simple Linear Regression model we want to draw a line between X and Y which estimates the relationship between X and Y.

But how do we find these coefficients? That’s the learning procedure. We can find these using different approaches. One is called **Ordinary Least Square Method** and other one is called **Gradient Descent Approach**. We will use Ordinary Least Square Method in Simple Linear Regression and Gradient Descent Approach in Multiple Linear Regression in post.

## Ordinary Least Square Method

Earlier in this post we discussed that we are going to approximate the relationship between X and Y to a line. Let’s say we have few inputs and outputs. And we plot these scatter points in 2D space, we will get something like the following image.

![](https://projects.ncsu.edu/labwrite/res/gh/2d-scatter-linreg.gif)

And you can see a line in the image. That’s what we are going to accomplish. And we want to minimize the error of our model. A good model will always have least error. We can find this line by reducing the error. The error of each point is the distance between line and that point. This is illustrated as follows.

![](https://docs.oracle.com/cd/E24693_01/datamine.11203/e16808/img/scatter_plot.gif)

And total error of this model is the sum of all errors of each point. ie.

  $$E = \sum_{i=1}^{m} e_i^2$$

  $$e_i$$ -Distance between line and i<sup>th</sup> point.

  $$m$$ - Total number of points

You might have noticed that we are squaring each of the distances. This is because, some points will be above the line and some points will be below the line. We can minimize the error in the model by minimizing E. And after the mathematics of minimizing E, we will get:

  $$\beta_1 = \frac{\sum_{i=1}^{m} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{m} (x_i - \bar{x})^2} $$

  $$\beta_0 = \bar{y} - \beta_1\bar{x}$$

In these equations $$\bar{x}$$ is the mean value of input variable **X** and $$\bar{y}$$ is the mean value of output variable **Y**.

Now we have the model. This method is called [Ordinary Least Square Method](https://www.wikiwand.com/en/Ordinary_least_squares). Now we will implement this model in Python.

## Implementation

We are going to use a dataset containing head size and brain weight of different people. This data set has other features. But, we will not use them in this model.. This dataset is available [here](https://github.com/gatij/Linear_Regression) in Github repo in csv file. Let’s start off by importing the data.



```python
#matplotlib inline for showing graphs in notebook as inline result
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#for setting size of figures
plt.rcParams['figure.figsize']=(20.0,10.0)

#Reading Data
data = pd.read_csv('headbrain.csv')
#print dimensions 
print(data.shape)
#take a look of data frame
data.head()

```

    (237, 4)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Gender</th>
      <th>Age Range</th>
      <th>Head Size(cm^3)</th>
      <th>Brain Weight(grams)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4512</td>
      <td>1530</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>3738</td>
      <td>1297</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>4261</td>
      <td>1335</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>3777</td>
      <td>1282</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>4177</td>
      <td>1590</td>
    </tr>
  </tbody>
</table>
</div>

As you can see there are 237 values in the training set. We will find a linear relationship between Head Size and Brain Weights. So, now we will get these variables.

```python
#intialization of independent and dependent variable vectors
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values
```

To find the values β1 and β0, we will need mean of X and Y. We will find these and the coeffients.


```python
# Mean of X and Y
mean_X = np.mean(X)
mean_Y = np.mean(Y)

# Total number of values
m = len(X)

# Using the formula to calculate b0 and b1
numerator = 0
denominator = 0
for i in range(m):
    numerator += (X[i] - mean_X) * (Y[i] - mean_Y)
    denominator += (X[i] - mean_X) * (X[i] - mean_X)
b1 = numerator/denominator
b0 = mean_Y - (b1 * mean_X)

# Print coefficients (Also called weights)
print(b1,b0)
```

    0.26342933948939945 325.57342104944223
    
There we have our coefficients.

$$Brain Weight = 325.573421049 + 0.263429339489 * Head Size$$

That is our linear model.

Now we will see this graphically.

```python
# Plotting X-Y graph and regression line
max_X=np.max(X)+100
min_X=np.min(X)-100

#Calculating line values x and y
#linspace() Return evenly spaced numbers over a specified interval
x = np.linspace(min_X,max_X,1000)
y = b0 + b1 * x

# Ploting Line
plt.plot(x, y, color='#58b970', label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y, c='#ef5423', label='Scatter Plot')

plt.xlabel('Head Size (cm^3)')
plt.ylabel('Brain Weight (grams)')
plt.legend()
plt.show()
```

![png](/img/in-post/post-Linear-Regression/simpleL_R_3_0.png)

This model is not so bad. But we need to find how good is our model. There are many methods to evaluate models. We will use **Root Mean Squared Error** and **Coefficient of Determination($$R^2$$ Score)**.

Root Mean Squared Error is the square root of sum of all errors divided by number of values, or Mathematically,

  $$RMSE = \sqrt{\sum_{i=1}^{m} \frac{1}{m} (\hat{y_i} - y_i)^2}$$

Here $$\hat{y_i}$$ is the i<sup>th</sup> predicted output values. Now we will find RMSE.



```python
#Finding how is our model calculating root mean square error
# lower the rmse value to 1 better the regression model (infact rmse is minimum here) 
rmse = 0
for i in range(m):
    y_pred = (Y[i]-(b0 + b1*X[i]))
    rmse += (y_pred * y_pred)
#Root mean square error
rmse=np.sqrt(rmse/m)    
print(rmse)    
```

    72.1206213783709
    
Now we will find $$R^2$$ score. $$R^2$$ is defined as follows,

$$SS_t = \sum_{i=1}^{m} (y_i - \bar{y})^2$$

$$SS_r = \sum_{i=1}^{m} (y_i - \hat{y_i})^2$$

$$R^2 \equiv 1 - \frac{SS_r}{SS_t}$$

$$SS_t$$ is the total sum of squares and $$SS_r$$ is the total sum of squares of residuals.

$$R^2$$ Score usually range from 0 to 1. It will also become negative if the model is completely wrong. Now we will find $$R^2$$ Score.



```python
# SS_t is the total sum of squares and SS_r is the total sum of squares of residuals. 
# closer the r^2 value to 1 better the regression model 
ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * X[i]
    ss_t += (Y[i] - mean_Y) ** 2
    ss_r += (Y[i] - y_pred) ** 2
r2 = 1 - (ss_r/ss_t)
print(r2)
```

    0.6393117199570003
    
0.63 is not so bad. Now we have implemented Simple Linear Regression Model using Ordinary Least Square Method. Now we will see how to implement the same model using a Machine Learning Library called [scikit-learn](http://scikit-learn.org/)


## The scikit-learn approach

[scikit-learn](http://scikit-learn.org/) is simple machine learning library in Python. Building Machine Learning models are very easy using scikit-learn. Let’s see how we can build this Simple Linear Regression Model using scikit-learn.



```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

m=len(X)
# Cannot use Rank 1 matrix in scikit learn
X = X.reshape((m, 1))
# Creating Model
reg = LinearRegression()
# Fitting training data
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

# Calculating RMSE and R2 Score
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X, Y)
# Compare these results with simpleL_R results(our linear regression model) 
print(np.sqrt(mse))
print(r2_score)

```

    72.1206213783709
    0.639311719957

You can see that this exactly equal to model we built from scratch, but simpler and less code.

Let we move on to Multiple Linear Regression


## **Multiple Linear Regression**

Multiple Linear Regression is a type of Linear Regression when the input has multiple features(variables).

## Model Representation

Similar to Simple Linear Regression, we have input variable(**X**) and output variable(**Y**). But the input variable has **n** features. Therefore, we can represent this linear model as follows;

$$Y = \beta_0 + \beta_1x_1 + \beta_1x_2 + … + \beta_nx_n$$

$$x_i$$ is the i<sup>th</sup> feature in input variable. By introducing $$x_0$$=1, we can rewrite this equation.

$$Y = \beta_0x_0 + \beta_1x_1 + \beta_1x_2 + … + \beta_nx_n$$

$$x_0 = 1$$

Now we can convert this eqaution to matrix form.

$$Y = \beta^TX$$

Where,

$$\beta = \begin{bmatrix}\beta_0 & \beta_1 & \beta_2 & .. & \beta_n\end{bmatrix}^T$$

and

$$X = \begin{bmatrix}x_0 & x_1 & x_2 & .. & x_n\end{bmatrix}^T$$

We have to define the cost of the model. Cost bascially gives the error in our model. Y in above equation is the our hypothesis(approximation). We are going to define it as our hypothesis function.

$$h_\beta(x) = \beta^Tx$$

And the cost is,

$$J(\beta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\beta(x^{\textrm{(i)}}) - y^{\textrm{(i)}})^2$$

By minimizing this cost function, we can get find β. We use **Gradient Descent** for this.

## Gradient Descent

Gradient Descent is an optimization algorithm. We will optimize our cost function using Gradient Descent Algorithm.

### Step 1

Initialize values $$β_0$$, $$β_1$$,…, $$β_n$$ with some value. In this case we will initialize with 0.

### Step 2

Iteratively update,

$$\beta_j := \beta_j - \alpha\frac{\partial}{\partial \beta_j} J(\beta)$$

until it converges.

This is the procedure. Here $$α$$ is the learning rate. This operation $$\frac{\partial}{\partial \beta_j} J(\beta)$$ means we are finding partial derivate of cost with respect to each $$β_j$$. This is called Gradient.

Read [this](https://math.stackexchange.com/questions/174270/what-exactly-is-the-difference-between-a-derivative-and-a-total-derivative) if you are unfamiliar with partial derivatives.

In step 2 we are changing the values of $$β_j$$ in a direction in which it reduces our cost function. And Gradient gives the direction in which we want to move. Finally we will reach the minima of our cost function. But we don’t want to change values of $$β_j$$ drastically, because we might miss the minima. That’s why we need learning rate.

![](http://www.xpertup.com/wp-content/uploads/2018/05/1-1.gif)

The above animation illustrates the Gradient Descent method.

But we still didn’t find the value of $$\frac{\partial}{\partial \beta_j} J(\beta)$$. After we applying the mathematics. The step 2 becomes.

$$\beta_j := \beta_j - \alpha\frac{1}{m}\sum_{i=1}^m (h_\beta(x^{(i)})-y^{(i)})x_{j}^{(i)}$$

We iteratively change values of $$β_j$$ according to above equation. This particular method is called **Batch Gradient Descent**.

## Implementation

Let’s try to implement this in Python. This looks like a long procedure. But the implementation is comparitively easy since we will vectorize all the equations. If you are unfamiliar with vectorization, read this [post](https://www.datascience.com/blog/straightening-loops-how-to-vectorize-data-aggregation-with-pandas-and-numpy/)

<div>
  <script src="https://gist.github.com/gatij/e3e36c64a7f6c9dec0b4a70f464b4ac6.js"></script>
</div>
