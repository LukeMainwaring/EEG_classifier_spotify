import pandas as pd
import numpy as np
import json

# numpy gives us an unnecessary warning when dividing a non-integer on line 76
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


''' Create a custom class to represent EEG data from a csv file. '''

class EEGData(object):
    ''' We will use pandas to read in a csv file and create a DataFrame'''
    def __init__(self, csv_file):
        # initialize with all data present in CSV
        all_data = pd.read_csv(csv_file)
        self.data = all_data

    ''' Another magic method that allows us to add, or concatenate, two
	different EEGData objects into one. This may be helpful if we want
	to combine datasets from separate csv files into one dataset.'''
    def __add__(self, other):
    	l = [self.data, other.data]
    	result = pd.concat(l)
    	self.data = result
    	return self

    ''' This function requires the user to pick which labels he/she wants to
	compare for the classifier. It also does the bulk of the work in cleaning
	up the data object.'''
    def choose_labels(self, label1, label2):
	    labeled_data = self.data[(self.data.label == label1)
		 | (self.data.label == label2)]
	    self.data = labeled_data
		# After choosing labels, remove unnecessary columns
	    for col in self.data.columns:
		    if col != "raw_values" and col != "label" and col != "id":
			    self.data = self.data.drop(col, axis=1)
		
		# Data object must contain 3 columns: id, raw_values, and label
	    if (self.data.shape[1]) != 3:
		    raise Exception("Invalid Input. Missing parameters.")


    ''' Allows user to enter their specific ID so that the classifier is
	toned to each specific user. '''
    def choose_id(self, ID):
		# drops all rows that are not specified ID
	    self.data = self.data[self.data.id == ID]
		
		# convert raw eeg values from str to list
	    self.data.raw_values = self.data.raw_values.map(json.loads)

		# store data as numpy array
	    self.data = np.array(self.data)
		
    ''' Once data is in correct format, this function creates our X and y vectors. '''
    def vectors(self):

		''' first we turn the raw_inputs into a power spectra, since this is how
		real EEG recordings are read and gives us more detailed info for
		when we create our classifier '''
	    for row in self.data:
		    vec = row[1]
		    A = np.fft.fft(vec)
		    ps = np.abs(A)**2
		    ps = ps[:len(ps)/2]
		    temp_vec = []
		    index = 1
		    avg = 0
		    for freq in ps:
			    ''' average each successive 8 values, which signifies that the classifier
				more easily finds distinctions between each 4 seconds, rather than
				every 0.5 seconds of eeg recordings '''
			    if index % 8 == 0:
				    avg += freq
				    avg = avg / 8
					# divide by 100 so we can more easily scale later
				    avg = avg / 100
				    temp_vec.append(avg)
				    avg = 0
			    else:
				    avg += freq
			    index += 1
		    row[1] = np.array(temp_vec)

		# create X (feature vector) and y (target)
	    X = []
	    y = []
		# arbitrarily pick first label as 1, assign other as 0
		# in this case, relaxation is 1
	    label_one = self.data[0][2]
	    for person in self.data:
		    if person[2] == label_one:
			    y.append(1)
			    X.append(person[1])
		    else:
			    y.append(0)
			    X.append(person[1])
	    return X, y


def main():
    # eeg_data = EEGData("eeg-data.csv")
    # eeg_data.choose_labels("music", "relax")
    # eeg_data.choose_id(1)
    # X, y = eeg_data.vectors()
    pass
    

if __name__ == "__main__":
	main()
