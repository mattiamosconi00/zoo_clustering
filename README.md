# zoo_clustering
 Python clustering project using a qualitative Kaggle dataset
The project aims at assessing some of the most known clustering algorithms by applying them to a simple categorical dataset. The dataset can be downloaded at https://www.kaggle.com/datasets/uciml/zoo-animal-classification. We’ll use “class.csv” as well as “zoo.csv”, which is the main dataset. 
The dataset contains 101 samples and 18 features. With the exception of “animal_name”, all features can be considered to be categorical features (“legs” included, even if it’s not a binary feature). Our target feature is “class_type”. We can associate class types with meaningful labels by looking at “class.csv”. For example, class_type 1 refers to mammals. 
We carried out an assessment of the clustering algorithm with two visualization methods and a scikit-learn metric (adjusted Rand index). 
