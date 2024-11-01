# Classification With Imbalanced Class

An imbalanced classification problem is where there is a highly skewed distribution towards a class. This leads to forcing the predictive models to an unrealistically high classification for majority class (overestimation) resulting very poor performance for minority class (underestimation). Therefore, this is a problem when the minority class is more important than the majority class. To resolve problematic classification of imbalanced classes,
resampling techniques are used to adjust the class distribution of training data (the ratio between the different classes) to feed more balanced data into predictive models; thereby creating a new transformed version of the training set with a different class distribution. Oversampling, Undersampling and combination of them are applied. To measure performance, K-fold cross validation is utilized but cross validation after resampling incorrectly leads to overoptimism and overfitting since the class distribution of the original data is different from the sampled training set. The correct approach is to
resample within K-fold cross validation. These techniques are discussed in details

