import numpy as np
from sklearn.linear_model import LogisticRegression 
from  sklearn.metrics import accuracy_score  
from sklearn.model_selection import train_test_split
from sklearn.utils import as_float_array 
import csv 
import math 
import collections 
keys = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,  
      'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11} 
genreStrings = [] 
allFeaturesMatrix = [] 
allFeaturesDict = {} 
masterGenreDict = {} 
songIdToNames = {}
genreArrayDict = {} 
genreToCoefficients = {} 
genreDict = {} 
misclassified = {}
misclassifiedPopSongs = []
topThreePredictions = {}
clusters1 = ['Pop', 'Electronic','Dance']
clusters2 = ['Hip-Hop', 'Rap', 'R&B']
clusters3 = ['Reggae','Soul', 'Ska']
clusters4 = ['World', 'Reggaeton','Movie']
clusters5 = ['Blues', 'Jazz', 'Soundtrack']
clusters6 = ['Country', 'Rock', 'Indie']
clusters7 = ['Alternative', 'Classical', 'Folk']
clusters8 = ['Opera', 'A Capella', 'Comedy']
  
allClusters = [clusters1, clusters2, clusters3, clusters4, clusters5, clusters6, clusters7, clusters8]
songToCluster = {}
misclassified = {}

 
songIdTo25Feature = {}

#songToGenre = {} 

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
            songName = ""
            nameID = ""
            for value in row :
                if currFeature ==0: 
                    if value not in genreStrings: 
                        genreStrings.append(value) 

                if (currFeature >= 4 and currFeature <= 8) or currFeature == 10 \
                    or currFeature == 11 or currFeature == 13 or currFeature == 14 \
                        or currFeature == 16: 
                        value = float(value)
                if currFeature == 15 : 
                    value = int(value[0])
                if currFeature ==2:
                    songName = value
                if currFeature == 3 : 
                    nameID = str(value)
                if currFeature == 9 : 
                    value = keys[value]
                if currFeature == 12 : 
                    value = 1 if value == "Major" else 0
                features.append(value)
                currFeature += 1
            songIdToNames[nameID] = songName
            allFeaturesDict[nameID] = features
            allFeaturesMatrix.append(features)
            rowNum += 1
        # store the row into our dict

def requests(name) :
    songRequests = collections.defaultdict(None)
    count = 0
    with open(name,'r') as file :
        for line in file: 
            if line == "\n":
                break
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
def makeGenreDict(filename, genreArray, cluster) : 
    with open(filename) as f: 
        line = csv.reader(f)
        rowNum = 0 
        for row in line: 
            if rowNum == 0 :  
                rowNum += 1 
                continue 
              # cast each value to a float 
              # features = [] 
            currFeature = 0 
            nameID = ""   
            genreY = 0 
            gname = ""  
            for value in row : 
                if currFeature ==0: 
                    gname = value 
                    if value in cluster: 
                        genreY = 1 
                if currFeature == 3 :  
                    nameID = str(value) 
                currFeature+=1 
            genreDict[nameID] = str(gname) 
#             print("keyyyy____")
#             print(nameID)
            genreArray.append(genreY) 
            rowNum += 1 
  #extract numerical features only, with the first value as 'id' 
def sigmoid(x): 
    return 1 / (1 + math.exp(-x)) 
def numericalFeaturesOnly(allFeatures, genreArray, toUse = []) :  
    songIds = []
    newFeatures = [] 
    
    if not toUse : 
        for feature in allFeatures : 
            songIds.append(feature[3]) 
            newFeatures.append(feature[4:]) 
    else :  
        for feature in allFeatures :  
            songIds.append(feature[3]) 
            feature = [feature[i] for i in range(len(feature)) if i in toUse] 
            newFeatures.append(feature)
            
    return newFeatures, songIds
def numericalFeaturesOnlyTree(allFeatures, genreArray, toUse = []) :  
    songIds = []
    newFeatures = [] 
    
    if not toUse : 
        for feature in allFeatures : 
            songIds.append(feature[3])
            genreArray.append(feature[0]) 
            newFeatures.append(feature[4:]) 
    else :  
        for feature in allFeatures :  
            songIds.append(feature[3]) 
            genreArray.append(feature[0])
            feature = [feature[i] for i in range(len(feature)) if i in toUse] 
            newFeatures.append(feature)
            
    return newFeatures, songIds 


def outRes2(x, y, songIds, toUse = [], average = True) : 
    necessaryDict = {} 
    totalSongs = 0 
    correct = 0  
#     print("x0::::::")
#     print(x[0])
    for i in range(len(x)) :  
#             print("\t\t" + allFeaturesDict[song][2] + " by " + allFeaturesDict[song][1])
        scores = [] 
        scoresDict = {} 
#         for song in songIds: 
#             j=0 
        for cluster in allClusters: 
            score =  np.dot(as_float_array(x[i], copy=True),genreToCoefficients[' '.join(cluster)][0])
#             print(as_float_array(x[i], copy=True))
#             print(genreToCoefficients[genreStrings[l]][0])
            sigmoidScore = sigmoid(score + genreToModel[' '.join(cluster)].intercept_)                            
            scores.append(sigmoidScore) 
            scoresDict[sigmoidScore] = ' '.join(cluster) 
        # print(genreDict[songIds[i]])
        # print(scoresDict)
        songIdTo25Feature[songIds[i]] = scores
        maxScore = max(scores)
        songToCluster[songIds[i]] = scoresDict[maxScore]
        # songname = songIds[i]
        # print(songIds[i])
        # 
        # topThreePredictions[songIds[i]] = []
        # scoresCopy = scores.copy()
        # for k in range(0, 3):
        #     max1 = float("-inf")
        #     for j in range(len(scoresCopy)):
        #         if scoresCopy[j] > max1:
        #             max1 = scoresCopy[j]
        #     scoresCopy.remove(max1)
        #     topThreePredictions[songIds[i]].append((max1, scoresDict[max1]))
        totalSongs+=1
        if(genreDict[songIds[i]] in songToCluster[songIds[i]].split(' ') or genreDict[songIds[i]]=='NULL'): 
            correct+=1 
        else:
            # print(genreDict[songIds[i]])
            # print(scoresDict[maxScore])
            if genreDict[songIds[i]] != 'NULL':
                if genreDict[songIds[i]] not in misclassified:
                    misclassified[genreDict[songIds[i]]] = collections.defaultdict(int)
                misclassified[genreDict[songIds[i]]][songToCluster[songIds[i]]]+=1
                
#             numIncorrect+=1 
#             if scoresDict[maxScore] not in misclassified:
#                 misclassified[scoresDict[maxScore]] = collections.defaultdict(int)
# #                 if genreDict[song] not in  misclassified[scoresDict[maxScore]]:
# #                     misclassified[scoresDict[maxScore]][genreDict[song]] = collections.defaultdict(int)
#             misclassified[scoresDict[maxScore]][y[i]]+=1


#     print("\n" ) 
    return correct, totalSongs
print("Reading File Data!\n"  ) 
# Our friend song interests.  
friends = requests('SongInput.txt') 
#populate matrix and dict with all features 
readMusicData('SpotifyFeatures.csv') 
print(genreStrings) 
# print(len(genreStrings)) 

  #Feature indices, beginning at 4 and ending at 16, inclusive. BTW, 12 features total.  
  #first example: all features 
  # desiredFeatures = [i for i in range(4,17)] 
  #second example: \  Danceability, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Loudness, and Tempo\   
def makeGenreDictTree(filename, genreArray, songs) : 
    with open(filename) as f: 
        line = csv.reader(f)
        rowNum = 0 
        for row in line: 
            if rowNum == 0 :  
                rowNum += 1 
                continue 
              # cast each value to a float 
              # features = [] 
            currFeature = 0 
            nameID = ""   
            gname = ""  
            for value in row : 
                if currFeature ==0: 
                    gname = value 
                if currFeature == 3 :  
                    nameID = str(value) 
                currFeature+=1 
            # genreDict[nameID] = str(gname) 
            if(nameID not in songs):
                genreArray.append(gname) 
            rowNum += 1 
from sklearn.model_selection import KFold
kf = KFold(n_splits=7, shuffle=True)
desFeat2 = [4, 5, 7, 8, 10, 11, 13, 14, 16] 

totalCorrect = 0
total = 0
genreToModel = {}
for train_index, test_index in kf.split(allFeaturesMatrix):
    print(train_index)
    X_train, X_test = [allFeaturesMatrix[j] for j in train_index], [allFeaturesMatrix[j] for j in test_index]
#     songIdsTrain = [songIds[j] for j in train_index]
    print(len(allFeaturesMatrix))
    print("SONG ID")
    # print(songIds[1])
    # songIdsTest = [songIds[j] for j in test_index]
    print(len(genreStrings))
    #     X_train, X_test = allFeaturesMatrix[train_index], allFeaturesMatrix[test_index]
    for cluster in allClusters: 
        mdl = LogisticRegression() 
        genreArray = [] 
    #     genreDict = {} 
        # print(genreStrings[i]) 
        
        makeGenreDict("SpotifyFeatures.csv" , genreArray, cluster) 
        # print(genreDict)
        y_train, y_test = [genreArray[j] for j in train_index], [genreArray[j] for j in test_index]
#         y_train, y_test = genreArray[train_index], genreArray[test_index]
    #     X_train, X_test, y_train, y_test = train_test_split(numericalFeatures, genreArray,test_size=0.005)
    #     print(genreArray) 
        # print(X_train[1])
        
        numericalFeatures,  songIds = numericalFeaturesOnly(X_train, desFeat2) 
        # print(numericalFeatures[1])
        fittedmdl = mdl.fit(numericalFeatures, y_train) 
    #     fittedmdl = mdl.fit(X_train, y_train) 
    #     genreToTest[genreStrings[i]] = (X_test, y_test)
        # print(genreStrings[i]) 
        # print(sum(genreArray)) 
    #     print(fittedmdl.coef_) 
        floats = as_float_array(fittedmdl.coef_, copy=True) 
        genreToCoefficients[' '.join(cluster)] = floats 
        genreArrayDict[' '.join(cluster)] = genreArray 
        masterGenreDict[' '.join(cluster)] = genreDict 
        genreToModel[' '.join(cluster)] = fittedmdl 
        numIncorrectClass = 0 
    print("done with one fold")
    # print(genreToModel)
    totalClassified = 0 
    # print(X_test)
    # numInc, totalUpdate = outRes2(X_test, y_test)
    numericalFeaturesTest, songIdsTest = numericalFeaturesOnly(X_test, desFeat2) 
    # print(numericalFeaturesTest[0])
    # print(len(y_test))
    corr, totalUpdate = outRes2(numericalFeaturesTest, y_test, songIdsTest)
    totalCorrect +=corr
    total += totalUpdate
print("Correct")
print(totalCorrect)
print("Total")
print(total)
print("accuracy")
if(total != 0):
    print(float(totalCorrect)/float(total))
print(misclassified)
# print("top three prediction:")
# print(topThreePredictions)
from sklearn import tree
desFeat3 = [4, 5, 8, 10, 11, 13, 14, 16] 
clusterToSongs = {}
for cluster in allClusters:
    clusterToSongs[' '.join(cluster)] = []
    for song in songToCluster:
        if songToCluster[song] == ' '.join(cluster):
            clusterToSongs[' '.join(cluster)].append(song)
# print(clusterToSongs)
songIdToPredictedGenre = {}
correct = 0
total = 0
for cluster in allClusters:
    arr = clusterToSongs[' '.join(cluster)]
    if(len(arr)==0): 
        print(cluster)
        continue
    
    kf2 = KFold(n_splits=4, shuffle=True)
    for train_index, test_index in kf2.split(arr):
        songsTrain, songsPredict = [arr[j] for j in train_index], [arr[j] for j in test_index]
        trainMatrix = []
        testMatrix = []
        songIds = []
        for song in songsTrain:
            trainMatrix.append(allFeaturesDict[song])
        for song in songsPredict:
            testMatrix.append(allFeaturesDict[song])
            songIds.append(song)
        clf = tree.DecisionTreeClassifier(max_depth=4, max_leaf_nodes = 25)
        genreArray1 = []
        numeric, hdhdh = numericalFeaturesOnlyTree(trainMatrix, genreArray1, desFeat3)
        # genreArray = []
        # makeGenreDictTree("SpotifyFeatures.csv", genreArray, cluster, songsTrain)
        # print(numeric)
        # print(genreArray1)

        clf = clf.fit(numeric, genreArray1)
        # y= numericalFeaturesOnlyTree([allFeaturesDict[song]], desFeat3)
        genreArray2 = []
        numericTest, songId2 = numericalFeaturesOnlyTree(testMatrix, genreArray2, desFeat3)
        y_pred=clf.predict(numericTest)
        # print(y_pred)
        for i in range(0, len(y_pred)):
            if y_pred[i] == genreDict[songIds[i]] or genreDict[songIds[i]]=='NULL':
                correct+=1
            total +=1
            songIdToPredictedGenre[songId2[i]] = y_pred[i]

print("num correct")
print(correct)
print("total")
print(total)
print("accruacy")
print(float(correct)/float(total))





with open('SpotifyFeatures.csv','r') as csvinput:
    with open('outputclustered.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = next(reader)
        row.append('Genre Prediction')
        all.append(row)

        for row in reader:
            row.append(songIdToPredictedGenre[row[3]])
            all.append(row)

        writer.writerows(all)
                                  