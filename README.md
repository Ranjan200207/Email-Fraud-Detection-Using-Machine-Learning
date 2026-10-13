# Email-Fraud-Detection-Using-Machine-Learning

**Project description**

This project explores how machine learning can classify text messages as **spam** or **ham** (legitimate messages). I used a dataset of 5,572 labeled SMS messages, including 4,825 ham and 747 spam messages. The class imbalance made it important to check spam-specific precision and recall in addition to overall accuracy.

I prepared the data in Python using Pandas by removing unused columns, mapping the labels to numeric values, and calculating message length for exploratory analysis. For text preprocessing, I removed punctuation and converted messages to lowercase. I then used scikit-learn’s **CountVectorizer** to turn message text into word-count features, and split the data into training and test sets using an 80/20 split.

I trained and compared four classifiers: **Multinomial Naive Bayes, Logistic Regression, Random Forest, and Support Vector Machine (SVM)**. On the notebook’s test split of 1,115 messages, Multinomial Naive Bayes achieved the best reported result: **98.65% accuracy**, with **99% precision, 90% recall, and a 95% F1 score for spam**. Logistic Regression reached 98.30% accuracy and 87% spam recall; Random Forest and SVM had lower spam recall on this split.

The results suggest that the Naive Bayes model was effective at identifying spam in this dataset, while the 90% recall also shows that some spam messages were missed. These figures describe one test split, so they should not be treated as proof of performance on new messages. Further evaluation could use a stratified split, a full preprocessing-and-model pipeline during cross-validation, and testing on a separate dataset. The notebook uses CountVectorizer; its comments mention stop-word removal, but that step is not implemented.
