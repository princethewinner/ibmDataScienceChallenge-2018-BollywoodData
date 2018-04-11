import numpy as np
import pickle as pkl

fileName = 'male_adjverb.csv'

with open(fileName, 'r') as fid:
    fileContent = fid.read()

fileContentLineByLine = fileContent.split('\n')

yearWiseWords = {}

for i in fileContentLineByLine[:-1]:
    splittedLine = map(lambda x: x.rstrip(']').lstrip('['), i.split(','))
    wordSet = set()
    year = int(splittedLine[0].strip(' '))
    for word in splittedLine[1:]:
        _word = word.strip(' ')
        if len(_word) > 0:
            wordSet |= set([_word])

    if year in yearWiseWords.keys():
        yearWiseWords[year] = list(wordSet | set(yearWiseWords[year]))
    else:
        yearWiseWords[year] = list(wordSet)

print(yearWiseWords)

with open('processed/{}'.format(fileName), 'w') as fid:
    for key in yearWiseWords.keys():
        fid.write('{},{}\n'.format(key, ','.join(yearWiseWords[key])))

with open('processed/{}.pkl'.format(fileName), 'w') as fid:
    pkl.dump(yearWiseWords, fid)
