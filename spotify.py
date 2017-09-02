''' This program includes aspects from the rest of the project. The Spotify
program begins by creating a classifier from the initial dataset we've been
using, and then using the classifier to predict whether or not the input data
is relaxing. Since I have no ability to generate EEG data, I have hardcoded
test_data for the predictions. If our classifier believes there's greater than
75 percent chance the brain waves suggest relaxation, then it adds the song to
a relaxation playlist, which must already be in the user's Spotify account.
The program takes in from the command line, in this order, the userID, Spotify
username, their relaxation playlistID, and the songID. Since these requests
require authentication, I manually entered my Spotify credentials so that I
could successfully add songs. '''


from clean_data import *
from classifier import *
import sys
import spotipy
import spotipy.util as util

''' Given a csv file of EEG recordings from a certain song, we first use
our classifier to classify the song as music or relaxation. If it is
relaxation we return true'''


def classify_song(clf, X, y, test_data):
    X = preprocessing.scale(X)
    clf.fit(X, y)

    predicted = clf.predict(test_data)
	# relax has value 1, so if classifier predicts that user is relaxing
	# based on input, then we return true
    if predicted.mean() > 0.75:
	    return True 
    else:
	    return False


''' If classifier predicts that this
song makes user feel as if he/she is deeply relaxed, then it will add the
song to a relaxation playlist.'''
def add_to_playlist():
	username = input('Enter your Spotify username: ')
	if len(sys.argv) > 2:
		playlist_id = sys.argv[1]
		track_id = sys.argv[2:]

	else:
	    sys.exit()

	scope = 'playlist-modify-public'
	# credentials received from spotify development account
	token = util.prompt_for_user_token(username, scope,
		client_id='YOUR_CLIENT_ID',
		client_secret='YOUR_CLIENT_SECRET',
		redirect_uri='http://localhost:8888/callback')

	if token:
	    sp = spotipy.Spotify(auth=token)
	    sp.trace = False
	    sp.user_playlist_add_tracks(username, playlist_id, track_id)
	    print("Successfully added song to relaxation playlist!")
	else:
	    print("Can't get token for", username)



def main():
    eeg_data = EEGData("eeg-data.csv")
    eeg_data.choose_labels("music", "relax")
    subject_id = input('Enter your ID number: ')
    subject_id = int(subject_id)
    eeg_data.choose_id(subject_id)
    X, y = eeg_data.vectors()
    clf = create_classifier(X, y)
    # now that we have the initial classifier, test it on new file
    test_data = X[:5]
    if classify_song(clf, X, y, test_data):
    	add_to_playlist()
    else:
    	print("Song was not considered relaxing and could not be added.")
    

if __name__ == "__main__":
	main()
