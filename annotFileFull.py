import pickle as pkl
import nltk

def getID(statement, maleWordList, femaleWordList):
    words = nltk.word_tokenize(statement.lower())
    retVals = {}
    for word in words:
        pMale = word in maleWordList
        pFemale = word in femaleWordList
        if pMale and pFemale:
            retVals[word] = 2
        elif pFemale:
            retVals[word] = 1
        elif pMale:
            retVals[word] = 0
    _c = statement.split(' ')
    return retVals, _c

movieName = 'yeh_to_kamaal_ho_gaya'
inFileName = 'movies/full_desc/{}'.format(movieName)
outputFile = 'movies/full_desc_annotated/{}'.format(movieName)

maleDictFile = 'male_adjverb.csv.pkl'
femaleDictFile = 'female_adjverb.csv.pkl'

femaleSynonymsFile = 'synonyms_of_female.txt'
maleSynonymsFile = 'synonyms_of_male.txt'

with open(maleDictFile, 'r') as fid:
    maleDict = pkl.load(fid)

with open(femaleDictFile, 'r') as fid:
    femaleDict = pkl.load(fid)

with open(maleSynonymsFile, 'r') as fid:
    content = fid.read()
content = content.split('\n')
content = set([x for x in content if len(x) > 0])

maleWords = set()
for key in maleDict.keys():
    maleWords |= set(maleDict[key])
print(len(maleWords))
maleWords |= content
print(len(maleWords))
maleWords = list(maleWords)

with open(femaleSynonymsFile, 'r') as fid:
    content = fid.read()
content = content.split('\n')
content = set([x for x in content if len(x) > 0])

femaleWords = set()
for key in femaleDict.keys():
    femaleWords |= set(femaleDict[key])
print(len(femaleWords))
femaleWords |= content
femaleWords = list(femaleWords)
print(len(femaleWords))
print('female' in femaleWords)


with open(inFileName, 'r') as ifid:
    with open(outputFile, 'w') as ofid:
        content = ifid.read()
        content = content.split('\n')
        for c in content:
            if len(c) > 0:
                ids, _c = getID(c, maleWords, femaleWords)
                #print(ids)
                ofid.write('{}\n'.format(_c))
                for key in ids.keys():
                    ofid.write('{} {}\n'.format(ids[key], key))
                ofid.write('\n')
