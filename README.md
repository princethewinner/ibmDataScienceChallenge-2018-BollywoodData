# ibmDataScienceChallenge-2018-BollywoodData

Step1: (Data Collection)
Download Bollywood-Dataset from
    https://github.com/BollywoodData/Bollywood-Data

    For the analysis wikipedia-data was used

Step2: (Data Collection)
Download GloVe word vectors from
    https://github.com/stanfordnlp/GloVe

    For the analysis "Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download): glove.6B.zip" was used.

Step3: (Data Processing)
    Process the adjverb file using the present in the wikipedia-data using script processAdjverb.py (for both male and female). This script will generate a pickle file which required for further processing of the data.

Step4: (Data Processing)
    Gather male and female synonyms. (Provided in the file synonyms_of_female.txt and Synonyms_of_male.txt)

Step5: (Data Processing)
    Seperate movie description of each movie from file wikipedia-data/coref_plot.csv. And save each movie in the seperate file.

Step6: (Data Processing)
    Replace all names with the corresponding gender.
    Replace all very specific words to bollywood with there corresponding noun. For example: "A village name Rampur." change this to "A village name noun". or "A village name place" or "A village name village". (This processing is needed to find the word hit in GloVe word dictionary.
    Translate or remove all the sentences written in Hindi.

Step7: (Data annotation)
    Use human perception to annotate the whole plot as male/female/neutral. male->0, female->1, neutral->2
    Use the following format for annotation.<br/>
    <annotations> <movie_description><br/>
    0 This is a male oriented movie.<br/>
    1 This is a female oriented movie.<br/>
    2 This is a gender neutral movie.<br/>

Step8: (Data annotation)
    Pass each movie description file resulted in step7 to the 'annotFileFull.py' for annotation of each know words based on the information available in adjverb file and synonyms file.
    ##ToDo : This script only annotates the adjectives and verbs from adjverb files and nouns from synonyms files. This script can be extended to annotate gender neutral words. This will improve the information content of the trees generated in the later steps.

Step9: (Data annotation)
    Based on the annotation of full movie description (Output of Step7), do a stratify sampling of movies to create train and validation sets. (Creation of test set will be described later).

Step10: (Data annotation)
    $var=['Train', 'Validation']
    for each movie in $var, split the output of that movie from step6 into lines and club them into single file.
    This step will create training and testing file.

Step11: (Data annotation)
    Pass each file resulted in Step10 to 'annotFileLineByLine.py'. This script will annotate each line with 0/1/2 based on the word frequecny of male and female oriented words and also annotate the male and female words seperatly.

Step12: (Tree Creation)
    Download coreNlp toolbox from https://github.com/stanfordnlp/CoreNLP. Maven-Build was used to create the .jar file. We will need https://github.com/stanfordnlp/CoreNLP/blob/master/src/edu/stanford/nlp/sentiment/BuildBinarizedDataset.java script for cration of trees from the annotated data. Jar created in this step can be used with the pre-setup stanford-parser available at https://nlp.stanford.edu/software/stanford-parser-full-2018-02-27.zip. Replace stanford-parser.jar from the stanford-parser-full-2018-02-27 folder with the stanford-corenlp-<build-version>.jar and rename this file to stanford-parser.jar.

Step13: (Tree Creation)
    Use script 'getTree.sh' to create trees for the output of Step8 and step11.
    Tree created with the output of Step11 will serve as the training and validation input to the Recursive-NN.
    Tree created with the output of Step8 will serve as the testing data.

Step14: (Training)
    Copy training trees to Recursive-NN/trees folder. For training tree name it train.txt, For validation tree name it dev.txt.
    Set training parameters in run.sh file and run it. It will create the model file in the models file.

    Structure implemented in this code is displayed in the poster.

    'wordMap.bin' contains words we used for training. As the dataset increases this file needs to be changed. Format of this file is as follows
    <word>,<indexNo>. This file is used to map the word in the dictionary to its vector representations.
    'wordMapEmbedding.txt'. This file contains the vector representations of words.

    Initial code was taken from https://github.com/cerberusd/cs224d-solutions/tree/master/assignment-3. Some modification were carried out on top of this code to suit our needs.

Step15: (Testing)
    Trees generated from the output of Step8 will be used for the testing set. Move all the trees in the Recursive-NN/trees folder and set the relevant parameters in the test.sh file and run it.

    Testing phase will dump extracted embedding for each test sample in the file. (It will be extracted embedding of the movie if whole movie script was provided as input).

Step16: (Testing)
    Use the extracted embedding from the Step15 and get Male and Female word embedding from GloVe word vectors and compute the angle between them. Movie will be oriented towards the word with the lesser angle.

Step17: (Testing)
    Movie orientation computed in the Step16 can be validated with the Movie Centrality information available in the wikipedia-data. We used the average centrality from the files female_centrality.csv and male_centrality.csv and results are documented in the poster. If movie centrality information spread accross multiple characters of a gender, Average was taken and the compared with the other gender. Higher the centrality, higher the movie bias towards the gender.




