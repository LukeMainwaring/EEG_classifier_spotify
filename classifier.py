from sklearn import preprocessing
from sklearn import datasets
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score
from clean_data import *


def create_classifier(X, y):
	''' This standardizes the dataset. This is important and increases the
	accuracy, since EEG recordings are fairly volatile and the features
	between different labels do not match up perfectly. '''
	x = preprocessing.scale(X)
	''' I tested a variety of classifiers in scikit-learn: KNN, Decision
	Tree, etc. and after computing overall accuracy among all subjects, I
	found that svm's were the best choice. '''
	clf = svm.SVC()

	''' Some classifiers are very accurate, but some are not much better than
	chance. This is due to the fact that EEG recordings are not perfect
	predictors of what a person is exactly thinking, and there is a
	correlation between attention levels (given in original Excel doc) and
	low classifier accuracies. Thus, if the classifier does not succeed we
	raise an exception. '''
	if cross_val_score(clf, x, y).mean() < 0.65:
		raise Exception("In order to create a classifier, more data and/or" +
			" paying better attention is necessary.")
	return clf


''' Generator to sum up the accuracy scores for each succesful classifier'''
def true_score(n):
	score = 0
	i = 1
	while i <= n:
		eeg_data = EEGData("eeg-data.csv")
		eeg_data.choose_labels("music", "relax")
		eeg_data.choose_id(i)
		X, y = eeg_data.vectors()
		x = preprocessing.scale(X)
		clf = svm.SVC()		
		score = cross_val_score(clf, x, y).mean()
		if score > 0.65:
			yield score
		i += 1


''' Generator to count how many subjects had successful classifiers '''
def num_successes(n):
	count = 1
	i = 1
	while i <= n:
		eeg_data = EEGData("eeg-data.csv")
		eeg_data.choose_labels("music", "relax")
		eeg_data.choose_id(i)
		X, y = eeg_data.vectors()
		x = preprocessing.scale(X)
		clf = svm.SVC()		
		if cross_val_score(clf, x, y).mean() > 0.65:
			yield count
		i += 1


def main():
	'''
	eeg_data = EEGData("eeg-data.csv")
	eeg_data.choose_labels("music", "relax")
	eeg_data.choose_id(1)
	X, y = eeg_data.vectors()
	classifier = create_classifier(X, y)
	x = preprocessing.scale(X)
	print(cross_val_score(classifier, x, y).mean())
	'''
	# If we want the average of the successful scores, we can print this:
	# print(sum(true_score(30)) / sum(num_successes(30)))
	pass


if __name__ == "__main__":
    main()
