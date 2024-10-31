# Practical_Application_III
Jupytor Notebook created for Practical Application Assignment for Module 17 of AI/ML Certification Course


October 31, 2024 
Professional Certificate in ML/AI 
Practical Application III 
Kimberly Tulga

Read ME: Comparing Classifiers Using Bank Marketing Data

Overview

The dataset provides information about the direct marketing campaign by a Portuguese bank. The dataset has 41,188 instances with 20 attributes and target variable 'y' which is a binary outcome that provides whether the customer signs up for a long-term deposit account. There has been a previous study done by Sérgio Moro and Raul M. S. Laureano from the University of Lisbon who created a highly precise ML model. Figure one shows the ROC Curve of the model they created.  

<img width="392" alt="Screen Shot 2024-10-31 at 1 17 19 PM" src="https://github.com/user-attachments/assets/9ec1e76f-c134-4114-9096-948d00d59886">

Figure 1: ROC Curve for ML model by Moro and Laureano

In their work, Moro & Laureano concluded that the most effective attribute that helps to determine the success of the marketing campaign is the duration of the last call. However, it should be noted that the duration of thies call gets longer when customers sign up for a new account. Therefore, this attribute is not the predictor, but the byproduct of the success of the campaign. In my model, I dropped the column that contained the 'duration' attribute. The dataset that was run by Moro & Laureano greatly differs from the one I received. For example, my dataset had 41,188 instances vs. 79,354 for Moro & Laureano. My dataset had 20 attributes, while Moro & Laureano’s first model had 59 attributes. Furthermore, my data set did not have an attribute called 'Duration of the first call' which was listed as the 6th most important attribute for their prediction model. 

I ran four different prediction models, namely Logistic Regression, K-N Neighbors, Decision Tree, and Support Vector Machine (SVM), and compared their results. 

Round 1 – Initial Run

I dropped the highly controversial 'duration' column and encoded the rest of the model to a numeric type. I confirmed that the dataset was imbalanced given the nature of the marketing campaign. In my first run, I left each and every model with their default setting. Models ran pretty fast, however, they didn’t produce as confident models as the ones by Moro & Laureano. Table 1 shows the accuracy rate and the performance time of each of my models.

 Model                Train score  Test score  Average fit time
                                                         
KNN                     0.911139    0.894241          5.596900
Logistic Regression     0.900262    0.900651          0.105780
SVC                     0.897446    0.897737         68.543189
Decision Tree           0.995856    0.845586          0.241004

Table 1: Result from the Initial one


<img width="563" alt="Screen Shot 2024-10-31 at 4 45 25 PM" src="https://github.com/user-attachments/assets/02986343-3c0d-4fe5-a8aa-ef1fa1d69f81">

Figure 2: ROC Curve for the initial run. 


Logistic Regression seemed to be the best model both in accuracy and performance. Decision Tree had the best time but had the lowest accuracy. SVM was the most resource-intensive model, taking more than 685x the Logistic Regression Model.  




Round 2 – Improving the Model

I added a set of parameters and ran GridSearchCV models in hopes to improve accuracy on each of the models. Because parameters bring inherent complexity and need more resources to run, I wanted to reduce the number of features that would be added to the model. I ran a ridge model to understand which features were the most important from the original 19 features. The ridge model showed that ['emp.var.rate', 'euribor3m', 'nr.employed', 'cons.price.idx', 'pdays', 'contact', 'month'] where the most relevant attributes shown in Table 2.

 	  feature 	        coef
4 	emp.var.rate 	    -0.190642
7 	euribor3m 	      0.154268
8 	nr.employed 	    -0.105342
5 	cons.price.idx 	  0.080235
2 	pdays 	          -0.051252
15 	contact 	        -0.042479
16 	month 	          -0.030056
6 	cons.conf.idx 	  0.018913
18 	poutcome 	        0.018598
12 	default 	        -0.007027
10 	marital 	        0.006108
17 	day_of_week 	    0.005794
3 	previous 	        -0.005419
0 	age 	            0.005228
11 	education 	      0.004619
1 	campaign 	        -0.004216
13 	housing 	        -0.001970
9 	job 	            0.000965
14 	loan 	            0.000537

Table 2: Results from the Ridge Model showing the most relevant features

However, when I added  “L1” penalty regularization for those features it was evident that four of them, namely 'emp.var.rate', 'euribor3m', 'nr.employed', and 'cons.price.idx' were highly correlated to each other shown in Figure 3. Since all four of them were socio-economic indicators that were taken quarterly, irrelevant to each individual's situation, I decided to keep only one of the four. 

<img width="1021" alt="Screen Shot 2024-10-31 at 4 46 20 PM" src="https://github.com/user-attachments/assets/8b32789f-5f9a-4623-a08d-af97f520a405">

Figure 3: Adding “L1” penalty to the seven most relevant features from the Ridge Model

Using a correlation matrix, I decided to keep the five most highly correlated features to my target ‘y’, minus the three seasonal indicators I eliminated previously. The result for my third run is shown in Table 3 and Figure 4 shows their corresponding ROC Curve. 


Model                    Average fit time  Train score  Test score  Best Model Parameters                                                                 
                                                            
KNN                        117.012141     0.897119    0.897889   {'knn__n_neighbors': 19, 'knn__p': 1, 'knn__we... 
Logistic Regression         37.490206     0.900262    0.900651   {'poly__degree': 3}
SVC                       3149.510995     0.897735    0.897889   {'svc__kernel': 'rbf'} 
Decision Tree                4.609488     0.897373    0.897521   {'max_depth': 2, 'min_impurity_decrease': 0.01, 'min_samples_split': 0.1}

Table 3: Results from the third set of models

<img width="556" alt="Screen Shot 2024-10-31 at 4 48 21 PM" src="https://github.com/user-attachments/assets/83a94730-f823-42bb-a5f0-24d0da91efac">

Figure 4: ROC Curve for the third set of models 

Key findings

The most important feature to predict success was the outcome from the previous call, followed by the number of previous calls made shown in Figure 5. The success rate of the campaign substantially improves when a customer was previously called at least once, but drastically improves when they were called at least twice. The third important indicator was when the unemployment rate went up, consumers tended to sign up for long-term deposit accounts shown in Figure 6. However, the default rate of the individual is probably an inconclusive feature that could be eliminated since there were no recorded instances in which the default was “Yes”; there were only ‘Non-existent’ or ‘No’ instances. 



<img width="1011" alt="Screen Shot 2024-10-31 at 4 46 54 PM" src="https://github.com/user-attachments/assets/5f556f2e-8de0-446d-a093-e7122b46e7d8">

Figure 5: “L1” penalty curve from the third set of models. 


<img width="582" alt="Screen Shot 2024-10-31 at 4 49 14 PM" src="https://github.com/user-attachments/assets/37a56899-34e6-44ef-9a18-d82e01cdaf14">

Figure 6: Employment Variable Rate vs. Successful Marketing Instances


Recommendation

Persistency is a key. People tend to sign up for long-term deposit accounts, particularly when they perceive there is softening in the labor market.


Works Cited

Moro, S., & Laureano, R. M. S. (2011, September 30). USING DATA MINING FOR BANK DIRECT MARKETING: AN APPLICATION OF THE CRISP-DM METHODOLOGY. Core.ac.uk. https://core.ac.uk/display/55616194?utm_source=pdf&utm_medium=banner&utm_campaign=pdf-decoration-v1
