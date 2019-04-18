# [Udacity](https://in.udacity.com/)
## [Machine-Learning-Engineer-Nanodegree](https://in.udacity.com/course/machine-learning-engineer-nanodegree--nd009t?utm_expid=.N5BPzO0yTgeKxF0Hi-Khhg.0&utm_referrer=https:%2F%2Fwww.google.co.in%2F)

Machine learning represents a key evolution in the fields of computer science, data analysis, software engineering, and artificial intelligence.

This program will teach you how to become a machine learning engineer, and apply predictive models to massive data sets in fields like finance, healthcare, education, and more.

**Co-Created with**:

1. [Kaggle](https://www.kaggle.com/)

2. [Amazon Web Services](https://aws.amazon.com/)

## Projects

### [Titanic Survival Exploration](https://github.com/MANOJPATRA1991/Machine-Learning-Engineer-Nanodegree/tree/master/P1-Titanic%20Survival%20Exploration)

In this practice project, we will create decision functions that attempt to predict survival outcomes from the 1912 Titanic disaster based on each passenger’s features, such as sex and age. We will start with a simple algorithm and increase its complexity until we are able to accurately predict the outcomes for at least 80% of the passengers in the provided data. This project will introduce us to some of the concepts of machine learning as we start the Nanodegree program.

### [Predicting Boston Housing Prices](https://github.com/MANOJPATRA1991/Machine-Learning-Engineer-Nanodegree/tree/master/P2-Boston%20Housing)

In this project, we will evaluate the performance and predictive power of a model that has been trained and tested on data collected from homes in suburbs of Boston, Massachusetts. A model trained on this data that is seen as a good fit could then be used to make certain predictions about a home — in particular, its monetary value. This model would prove to be invaluable for someone like a real estate agent who could make use of such information on a daily basis.

Things learnt by completing this project:

* How to explore data and observe features.
* How to train and test models.
* How to identify potential problems, such as errors due to bias or variance.
* How to apply techniques to improve the model, such as cross-validation and grid search.

### [Spam Classifier](https://github.com/MANOJPATRA1991/Machine-Learning-Engineer-Nanodegree/tree/master/P3-Spam%20Classifier)

In this practice project, we create a model that can classify dataset SMS messages as spam or not spam, based on the training we give to the model. For this project we make use of 

* Naive-Bayes algorithm and 
* Bag of Words(BoW) concept which is a term used to specify the problems that have a 'bag of words' or a collection of text data that needs to be worked with.

Concepts learned:

* Bag of Words(BoW)
* Implementing Bag of Words in scikit-learn
* Implementation of Bayes Theorem
* Limitations of Bayes Theorem
* Implementation of Naive Bayes Algorithm
* Implementation of Naive Bayes using scikit-learn
* Major advantages of Naive Bayes classification algorithm

### [Finding Donors for Charity ML](https://github.com/MANOJPATRA1991/Machine-Learning-Engineer-Nanodegree/tree/master/P4-Finding%20Donors%20for%20Charity%20ML)

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. With nearly 15 million working Californians, CharityML has brought you on board to help build an algorithm to best identify potential donors and reduce overhead cost of sending mail. Your goal will be evaluate and optimize several different supervised learners to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent.

In this project, I employed three supervised algorithms of my choice, namely, **Adaboost**, **SVM** and **Naive Bayes** to accurately model individuals' income using data collected from the 1994 U.S. Census. I then chose **Adaboost** as the best candidate algorithm from preliminary results and further optimized this algorithm to best model the data. 

The goal was to construct a model that accurately predicts whether an individual makes more than $50,000. 

### [Customer Segment Identification](https://github.com/MANOJPATRA1991/Machine-Learning-Engineer-Nanodegree/tree/master/P11-Customer%20Segments)

In this project we will apply unsupervised learning techniques on product spending data collected for customers of a wholesale distributor in Lisbon, Portugal to identify customer segments hidden in the data. We will first explore the data by selecting a small subset to sample and determine if any product categories highly correlate with one another. Afterwards, we will preprocess the data by scaling each product category and then identifying (and removing) unwanted outliers. With the good, clean customer spending data, we will apply PCA transformations to the data and implement clustering algorithms to segment the transformed customer data. Finally, we will compare the segmentation found with an additional labeling and consider ways this information could assist the wholesale distributor with future service changes.

Things learned by completing this project:

* How to apply preprocessing techniques such as feature scaling and outlier detection.
* How to interpret data points that have been scaled, transformed, or reduced from PCA.
* How to analyze PCA dimensions and construct a new feature space.
* How to optimally cluster a set of data to find hidden patterns in a dataset.
* How to assess information given by cluster data and use it in a meaningful way.

### [Dog breed Classifier](https://github.com/MANOJPATRA1991/Machine-Learning-Engineer-Nanodegree/tree/master/P12-Dog%20Breed%20Classifier%20(CNN%20Project))

In this project which takes advantage of Convolutional Neural Networks (CNN), we will learn how to build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.  Given an image of a dog, our algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed.  

![Sample Output](https://github.com/MANOJPATRA1991/Machine-Learning-Engineer-Nanodegree/blob/master/P12-Dog%20Breed%20Classifier%20(CNN%20Project)/images/sample_dog_output.png)

Along with exploring state-of-the-art CNN models for classification, we will make important design decisions about the user experience for our app.  Our goal is that by completing this lab, we understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline.  Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer. Our imperfect solution will nonetheless create a fun user experience.

### [Teaching a quadcopter how to fly](https://github.com/MANOJPATRA1991/Machine-Learning-Engineer-Nanodegree/tree/master/P18-Teaching%20a%20Quadcopter%20how%20to%20fly)

The Quadcopter or Quadrotor Helicopter is becoming an increasingly popular aircraft for both personal and professional use. Its maneuverability lends itself to many applications, from last-mile delivery to cinematography, from acrobatics to search-and-rescue.

Most quadcopters have 4 motors to provide thrust, although some other models with 6 or 8 motors are also sometimes referred to as quadcopters. Multiple points of thrust with the center of gravity in the middle improves stability and enables a variety of flying behaviors.

But it also comes at a price–the high complexity of controlling such an aircraft makes it almost impossible to manually control each individual motor's thrust. So, most commercial quadcopters try to simplify the flying controls by accepting a single thrust magnitude and yaw/pitch/roll controls, making it much more intuitive and fun.

The next step in this evolution is to enable quadcopters to autonomously achieve desired control behaviors such as takeoff and landing. We could design these controls with a classic approach (say, by implementing PID controllers). Or, we can use reinforcement learning to build agents that can learn these behaviors on their own. This is what we did in this project.

