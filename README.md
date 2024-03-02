Algorithmic Insights into Parkinson's 
Disease Prediction 
Rahul, Pooja Rana 
Lovely Professional University Phagwara, Punjab 
Abstract 
This research paper addresses the imperative need for accurate and timely diagnosis of 
Parkinson's disease (PD), a widespread neurodegenerative disorder causing substantial 
challenges for individuals and healthcare systems globally. Traditional diagnostic methods 
often fall short in sensitivity, prompting the exploration of advanced technologies, specifically 
machine learning (ML). The research employs K-Nearest Neighbors (KNN), Support Vector 
Machines (SVM) with polynomial and radial basis function (RBF) kernels, and Decision Tree 
algorithms to predict PD based on a diverse dataset. By harnessing ML's potential, the study 
aims to develop a robust predictive model, facilitating early PD identification for personalized 
interventions. The paper includes an in-depth exploration of the dataset, methodology, and 
evaluation metrics. Notably, the KNN algorithm demonstrates strong performance, 
emphasizing the significance of hyperparameter tuning in optimizing model accuracy. Overall, 
the research contributes valuable insights into the application of ML for Parkinson's disease 
diagnosis, holding promise for enhanced healthcare practices and outcomes. 
Keywords: Parkinson's Disease, Machine Learning, K-Nearest Neighbors (KNN), Support 
Vector Machines (SVM), Decision Tree Algorithms 
Introduction 
Parkinson's disease (PD) is a neurodegenerative disorder that affects millions of people worldwide, causing 
significant challenges to both patients and healthcare systems. Characterized by the progressive loss of 
dopaminergic neurons in the substantia nigra region of the brain, PD manifests with a variety of motor and non
motor symptoms, including tremors, bradykinesia, rigidity, and postural instability. 
The early and accurate diagnosis of Parkinson's disease is crucial for timely intervention and effective 
management of the condition. Traditional diagnostic methods rely heavily on clinical assessments, which may not 
be sufficiently sensitive or specific, leading to potential misdiagnoses. As the field of medical research continues 
to embrace technological advancements, machine learning (ML) has emerged as a promising tool for enhancing 
the accuracy and efficiency of disease diagnosis. 
This research endeavours to leverage machine learning techniques to predict the likelihood of Parkinson's disease 
based on a curated dataset. The dataset encompasses a diverse range of clinical and demographic features, offering 
a rich source of information for training and evaluating predictive models. By harnessing the power of ML 
algorithms, we aim to develop a robust predictive model that can aid in the early identification of Parkinson's 
disease, paving the way for more targeted and personalized medical interventions. 
The use of machine learning in medical diagnostics has shown remarkable success in various domains, and its 
application to Parkinson's disease detection holds the potential to revolutionize how we approach diagnosis and 
1 | P a g e 
treatment. This paper presents an in-depth exploration of the dataset, the methodology employed in model 
development, and the evaluation metrics used to assess the predictive performance. Additionally, we discuss the 
implications of our findings for the broader landscape of Parkinson's disease research and healthcare. 
Literature Review 
This study [1] focused on utilizing machine learning techniques for the detection of Parkinson's disease, 
specifically targeting speech impairments associated with the condition. Parkinson's disease, characterized by 
neurodegeneration and the loss of dopaminergic neurons, manifests in motor system disruptions. The paragraph 
underscores the prevalent speech impairments in Parkinson's patients and their significance for early detection. 
The study employs the Unified Parkinson Sickness Rating Scale (UPDRS) and conducts feature selection on an 
audio dataset obtained from the University of Oxford. Utilizing Pearson’s correlation coefficient, the research 
analyzes the relationships among the dataset features. A comparative evaluation of model-based (logistic 
regression) and model-free (XGBoost) approaches reveals that XGBoost outperforms logistic regression in 
classifying Parkinson's disease, achieving a higher accuracy of 96% compared to 79%. This study emphasizes the 
potential of machine learning algorithms, particularly XGBoost, in effectively classifying and detecting 
Parkinson's disease based on high-dimensional data. 
This study [2] discusses Parkinson's Disease (PD), highlighting its prevalence as the second most common age
related neurological disorder with motor and cognitive symptoms. Diagnosis is challenging due to similarities 
with other conditions, and visible symptoms typically emerge around the age of 50. While there is no cure for PD, 
medications can alleviate some symptoms, emphasizing the importance of early detection to prevent progression. 
The paragraph introduces a project aiming to detect PD using machine learning and deep learning approaches, 
focusing on voice signal features. The proposed model achieves high accuracy, with SVM at 95% and MLP at 
98.31%, surpassing previous works. The model has potential applications in reducing treatment costs, serving as 
an educational tool, and enhancing diagnostic capabilities for physicians. Future improvements are suggested to 
further enhance accuracy and scalability. 
This study [3] outlines the challenges in diagnosing Parkinson's disease (PD), emphasizing the subjectivity and 
potential misclassification associated with traditional diagnostic methods. It highlights the difficulty of detecting 
subtle movements and the often-overlooked non-motor symptoms in the early stages of PD. To address these 
challenges and enhance diagnostic procedures, the abstract discusses the implementation of machine learning 
methods for classifying PD, healthy controls, and patients with similar clinical presentations. The review 
investigates the aims, data sources, types of data, machine learning methods, and outcomes of these studies. The 
findings suggest a high potential for the integration of machine learning methods and novel biomarkers in clinical 
decision-making, promising more systematic and informed diagnosis of Parkinson's disease. 
Methodology 
In this research paper, the application of K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and 
Decision Tree algorithms is explored.  
Support Vector Machines (SVM) is a supervised machine learning algorithm used for classification and regression 
tasks. The fundamental concept of SVM involves finding the optimal hyperplane that maximally separates 
different classes in the feature space. It achieves this by identifying support vectors, which are data points closest 
to the decision boundary, and aims to maximize the margin between classes, providing a robust and effective 
means of classification. 
Decision Trees recursively split data based on features to create a tree-like structure, making decisions at each 
node. The splits optimize for information gain or purity, and the final leaf nodes represent predicted outcomes or 
class labels. Known for interpretability, Decision Trees are versatile for both classification and regression tasks. 
The fundamental concept underlying KNN involves predicting the class of a given data point by examining its k 
nearest neighbors within the feature space. The KNN algorithm follows a distinct methodology: 
• Training: 
2 | P a g e 
The algorithm starts by storing all available training examples. 
• Calculate Distance 
To predict the class of a new data point, the algorithm calculates the distance between that point and every point 
in the training set using Euclidean distance. 
Euclidean distance = √ ((xi – xnew)2 + (yi – ynew)2 ) 
Where, xi and yi are datapoints having known class and xnew and ynew new datapoint. 
• Identify Neighbors 
The algorithm then identifies the k training examples with the smallest distances to the new data point. 
• Majority Voting 
The algorithm assigns the class label that is most common among the k neighbors. This is typically done through 
majority voting. 
Flow diagram 
Parkinson’s 
disease Dataset 
Data 
Preprocessing 
Feature Selection 
Training 
data (80%) 
Splitting the data 
ML Models 
Testing Data 
(20%) 
Evaluate 
performance. 
Healthy 
people 
Data preprocessing 
Parkinson 
people 
StandardScaler is a preprocessing technique in scikit-learn, particularly useful when dealing with machine 
learning algorithms that assume features are normally distributed. StandardScaler standardizes features by 
removing the mean and scaling to unit variance. This process ensures that the transformed features have a mean 
of 0 and a standard deviation of 1.  
3 | P a g e 
Feature Selection 
In the context of feature selection, the correlation matrix serves as a valuable tool for examining the relationships 
between input features and the target output, in this case, the "status." By visualizing this correlation matrix 
through a heatmap, patterns in the correlation values become apparent.  
Upon analysis of the heatmap, it has been observed that certain features, namely "name," "MDVP:Fhi(HZ)," 
"NHR," and "DFA," exhibit a very low correlation with the output variable. This low correlation implies that these 
features may not significantly influence the output, and their removal from the dataset is considered.  
The decision to eliminate these features is grounded in the understanding that retaining only the most relevant 
features can enhance model interpretability, reduce computational complexity, and potentially improve the model's 
predictive performance. By iteratively refining the dataset through the removal of less impactful features, the 
feature selection process aims to streamline the modelling process and focus on the most informative variables 
for predicting the target output. 
Result 
Comparison Table 
In this model evaluation, Support Vector Machines (SVC) with polynomial and radial basis function (RBF) kernels 
show signs of overfitting, as seen in the decline of testing accuracy compared to training accuracy. The Decision 
Tree Classifier exhibits perfect accuracy on the training set but struggles to generalize, resulting in a notable 
decrease in testing accuracy. 
4 | P a g e 
The K-Nearest Neighbors (KNN) algorithm demonstrated strong classification performance in this study, 
achieving an impressive accuracy of 96.7% on the training dataset and maintaining a robust accuracy of 94.8% 
on the testing dataset. Hyperparameter tuning was conducted to optimize the choice of k ranging from 2 to the 
square root of the dataset length. 
From fig.  the model exhibited peak accuracies at k values of both 3 to 5, with the ultimate selection of k as 3 due 
to its consistently superior performance. This fine-tuned configuration resulted in a 94.8% accuracy on the testing 
dataset, highlighting the model's ability to generalize effectively to new instances. The study underscores the 
significance of meticulous hyperparameter tuning in enhancing the KNN model's predictive capabilities for 
optimal classification outcomes. 
5 | P a g e 
Conclusion 
In conclusion, this research paper has provided a comprehensive examination of various machine learning models 
applied to a specific problem. Challenges were observed in models such as Support Vector Machines (SVC) with 
polynomial and radial basis function (RBF) kernels, as well as the Decision Tree Classifier, which exhibited signs 
of overfitting, particularly reflected in the drop in testing accuracy. The linear SVC model performed reasonably 
well but still indicated some overfitting tendencies. However, K-Nearest Neighbors (KNN) algorithm 
demonstrated strong performance with 96.7% accuracy on the training data and 94.8% on the testing data. A for 
loop was employed to identify the optimal hyperparameter k with values of 3 to 5 yielding the highest accuracies, 
and k was chosen as 3 for its consistent superior performance. This fine-tuned configuration resulted in a robust 
94.8% accuracy on the testing dataset, emphasizing the effectiveness of thoughtful hyperparameter tuning in 
enhancing model accuracy and generalization. 
6 | P a g e 
References 
1. M.S. Roobini1 , Yaragundla Rajesh Kumar Reddy2 , Udayagiri Sushmanth Girish Royal3 , 
Amandeep Singh K4 , Babu.K5, 2022, Parkinson’s Disease Detection Using Machine Learning. 
2. Raya Alshammri,  Ghaida Alharbi,  Ebtisam Alharbi, and Ibrahim Almubark, 2023, Machine 
Learning approaches to identify Parkinson’s disease using voice signal features. 
3. Jie Meia*, Christian Desrosiersb and Johannes Frasnellia,c, Machine learning for the diagnosis 
of Parkinson’s disease: A systematic review 
4. Dr. Pooja Raundale, Chetan Thosar, Shardul Rane, Prediction of Parkinson’s disease and 
severity of the disease using Machine Learning and Deep Learning algorithm 
5. Yuqi Qiu, 2022,“Efficient Pre-diagnosis Approach for Parkinson’s Disease with Machine 
Learning 
6. Shrinidhi Kulkarni, Neenu George Kalayil, Jinu James, Sneha Parsewar and Revati Shriram, 
2020, Detection of Parkinson’s Disease through Smell Signatures 
7. Hanbin Zhang, Chen Song, Aditya Singh Rathore, Ming-Chun Huang, Yuan Zhang, Wenyao Xu, 
2020, Health Technologies towards Parkinson’s Disease Detection and Monitoring in Daily Life 
7 | P a g e 
