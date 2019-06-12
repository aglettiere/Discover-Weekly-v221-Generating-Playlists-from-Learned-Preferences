import numpy as np
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score 
import csv
import collections
keys = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}

allFeaturesMatrix = []
allFeaturesDict = {}
genreDict = {}
genreArray = []

def readMusicData(filename) :
    # open a file
    with open(filename) as f:
        line = csv.reader(f)
        # loop over each row in the file
        rowNum = 0
        for row in line:
            if rowNum == 0 : 
                rowNum += 1
                continue
            # cast each value to a float
            features = []
            currFeature = 0
            nameID = ""
            for value in row :
                if (currFeature >= 4 and currFeature <= 8) or currFeature == 10 \
                    or currFeature == 11 or currFeature == 13 or currFeature == 14 \
                        or currFeature == 16: 
                        value = float(value)
                if currFeature == 15 : 
                    value = int(value[0])
                if currFeature == 3 : 
                    nameID = str(value)
                if currFeature == 9 : 
                    value = keys[value]
                if currFeature == 12 : 
                    value = 1 if value == "Major" else 0
                features.append(value)
                currFeature += 1
            allFeaturesDict[nameID] = features
            allFeaturesMatrix.append(features)
            rowNum += 1
        # store the row into our dict

def requests(name) :
    songRequests = collections.defaultdict(None)
    count = 0
    with open(name,'r') as file :
        for line in file: 
            line = line.split(' = ')
            name = line[0]
            line[1] = line[1].replace('\"', "")
            ids = line[1].split(", ")
            ids[0] = ids[0].replace('c(', "")
            ids[4] = ids[4].replace(')\n', "")
            songRequests[name] = ids
            count += 1
            if count == 51 : break
    return collections.OrderedDict(sorted(songRequests.items()))

def makeGenreDict(filename) :
    # open a file
    with open(filename) as f:
        line = csv.reader(f)
        # loop over each row in the file
        rowNum = 0
        for row in line:
            if rowNum == 0 : 
                rowNum += 1
                continue
            # cast each value to a float
            # features = []
            currFeature = 0
            nameID = ""
            genre = ""
            for value in row :
                if currFeature ==0:
                    genre = value
                if currFeature == 3 : 
                    nameID = str(value)
            genreDict[nameID] = genre
            genreArray.append(genre)
            rowNum += 1
#extract numerical features only, with the first value as 'id'
def numericalFeaturesOnly(allFeatures, toUse = []) : 
    newFeatures = []
    if not toUse :
        for feature in allFeatures :
            newFeatures.append(feature[4:])
    else : 
        for feature in allFeatures : 
            feature = [feature[i] for i in range(len(feature)) if i in toUse]
            newFeatures.append(feature)
    return newFeatures
# def net_input(theta, x):
#     # Computes the weighted sum of inputs
#     return np.dot(x, theta)

# def probability(theta, x):
#     # Returns the probability after passing through sigmoid
#     return sigmoid(net_input(theta, x))
# def sigmoid(x):
#     # Activation function used to map any real value between 0 and 1
#     return 1 / (1 + np.exp(-x))
def outputResults(model, toUse = [], average = True) :
    for person, songIds in friends.items() : 
        songs = []
        for song in songIds : 
            #Using all songs
            info = allFeaturesDict[song]
            if not toUse : songs.append(info[4:])
            else : songs.append([info[i] for i in range(len(info)) if i in toUse])
        
        
        print "Songs for my friend " + person
        print "\t Input songs: "
        for song in songIds : 
            print "\t\t" + allFeaturesDict[song][2] + " by " + allFeaturesDict[song][1]
        # print "\t Averaged features: "
        # #average of all the features!
        # for line in result : 
        #     for song in line : 
        #         print "\t\t" + allFeaturesMatrix[song][2] + " by " + allFeaturesMatrix[song][1]
        # print "\t Nearest neighbor to individual song: "
        predicted_classes = model.predict(songs)
        print predicted_classes
        totalSongs = 0
        #only the nearest neighbor that is not itself!
        # for line in result2 : 
        #     print "\t\t" + allFeaturesMatrix[line[1]][2] + " by " + allFeaturesMatrix[line[1]][1]
        print "\n"

print "Reading File Data!\n"
# Our friend song interests. 
friends = requests('SongInput.txt')
#populate matrix and dict with all features
readMusicData('SpotifyFeatures.csv')

#Feature indices, beginning at 4 and ending at 16, inclusive. BTW, 12 features total. 
#first example: all features
# desiredFeatures = [i for i in range(4,17)]
#second example: "Danceability, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Loudness, and Tempo"
desiredFeatures = [4,5,8,10,11,13,14,16]
#just the numerical features sans artist/titles/genre/id
numericalFeatures = numericalFeaturesOnly(allFeaturesMatrix, desiredFeatures)
model = LogisticRegression()
model.fit(allFeaturesMatrix, genreArray)
outputResults(model, desiredFeatures)

# predicted_classes = model.predict(X)
# accuracy = accuracy_score(genreArray.flatten(),predicted_classes)
# parameters = model.coef_


