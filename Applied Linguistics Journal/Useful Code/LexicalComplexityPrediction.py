import pandas as pd
import numpy as np
from scipy import stats
import re
import syllables
import seaborn as sns

import matplotlib.pyplot as plt
import nltk
from nltk.corpus import reuters, brown
import textstat

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from gensim.models import Word2Vec


singleWords = pd.read_csv("lcp_single_train.tsv", sep = '\t')
multiWords = pd.read_csv("lcp_multi_train.tsv", sep='\t')

# singleWords_test = pd.read_csv("lcp_single_test.tsv", sep = '\t')
#multiWords_test = pd.read_csv("lcp_multi_test.tsv", sep='\t', error_bad_lines=False, engine="python")
def cleanDataset(words):
    """
        A function to clean the datasets.

        :param words: the original datasets in the form of a dataframe
        :return: clean dataframe
    """
    words_clean = pd.DataFrame(columns=['id', 'corpus', 'sentence', 'token', 'complexity'])

    for ind in words.index:
        if 'bible' in words['sentence'][ind]:
            continue
        else:
            words_clean = words_clean.append({'id': words['id'][ind], 'corpus': words['corpus'][ind],
                                                          'sentence': words['sentence'][ind], 'token': words['token'][ind],
                                                          'complexity': words['complexity'][ind]}, ignore_index=True)

    words_clean = words_clean.dropna()

    return words_clean


def cleanDataset_test(words):
    """
        A function to clean the datasets.

        :param words: the original datasets in the form of a dataframe
        :return: clean dataframe
    """
    words_clean = pd.DataFrame(columns=['id', 'corpus', 'sentence', 'token'])

    for ind in words.index:
        if 'bible' in words['sentence'][ind]:
            continue
        else:
            words_clean = words_clean.append({'id': words['id'][ind], 'corpus': words['corpus'][ind],
                                              'sentence': words['sentence'][ind], 'token': words['token'][ind]},
                                              ignore_index=True)

    words_clean = words_clean.dropna()

    return words_clean



def cleanCorpusText(data):
    """
        This function cleans the raw text that has been collected from a corpus

        :param data: a list of sentences from the corpus
        :return: clean string without special characters
    """
    completeCorpus = ""

    for eachSentence in data:
        eachSentence = eachSentence.lower()
        eachSentence = re.sub(r'[^\w]', ' ', eachSentence)
        completeCorpus += eachSentence

    return completeCorpus




def getCorpusText():
    """
        Combines and returns raw text of a large corpus in the form of a list

        :return: list of strings that are sentences in the corpora
    """
    reutersList = [" ".join(reuters.words(fid)) for fid in reuters.fileids()]
    combinedList = [" ".join(brown.words(fid)) for fid in brown.fileids()]

    for eachString in reutersList:
        combinedList.append(eachString)

    return combinedList




def getSyllableCount(word):
    """
        A function to count syllables

        :param word: word whose syllables will be counted
        :return: number of syllables in the word
    """
    words = word.split(" ")
    count = 0

    for eachWord in words:
        count += textstat.syllable_count(eachWord)

    return count




def lengthCount(word):
    """
        Count length of word.

        :param word: word whose length will be counted
        :return: length of word
    """
    words = word.split(" ")
    count = 0
    for eachWord in words:
        count += len(eachWord)

    return count




def getBNCCorpusFreqList():
    """
        Cleaning and preparing the British National Corpus (BNC)
        frequency list to use it for adding frequencies to the
        tokens later.

        :return: clean dataframe of BNC frequency list
    """
    print("Inside BNC collect...")
    #return pd.DataFrame()
    wordFrequencyDf = pd.read_csv('1_1_all_fullalpha.csv',
                                  names=['word', 'pos', 'extra', 'frequency', 'range', 'dispersion'])

    wordFrequencyDf['word'] = wordFrequencyDf['word'].astype(str)
    #print(wordFrequencyDf)

    BAD_CHARS = ['.', '&', '(', ')', ';', '-', '%', '@', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    pat = '|'.join(['({})'.format(re.escape(c)) for c in BAD_CHARS])

    wordFrequencyDf = wordFrequencyDf[~wordFrequencyDf['word'].str.contains(pat)]
    wordFrequencyDf = wordFrequencyDf.drop(labels=['extra','pos','range','dispersion'] , axis=1)
    #print(wordFrequencyDf)

    wordFrequencyDictTemp = wordFrequencyDf.set_index('word').to_dict()

    wordFrequencyDict = {}

    for key in wordFrequencyDictTemp.keys():
        token = key.lower()
        if token not in wordFrequencyDict.keys():
            wordFrequencyDict.update({token : wordFrequencyDictTemp[key]})
        else:
            wordFrequencyDict[token] += wordFrequencyDictTemp[key]

    #print(wordFrequencyDict)


    return wordFrequencyDict





def tokenOccurence(token, allWords, wordFrequencyDict):
    """
        Finding out the number of times the token occured in the dataset. The function
        is universal for single words and multi-word expressions

        :param token: token that is being searched
        :param allWords: words from the Reuters and Brown corpora
        :param wordFrequencyDict: BNC word frequency list

        :return: the count of the token occurence
    """

    frequency = 0
    token = token.lower()

    #Convert brown and reuter corpora words to a frequency list
    allWordsDf = allWords.value_counts()
    allWordsDict = allWordsDf.to_dict()

    wordsInToken = token.split(" ")

    # if the token is a MWE
    if len(wordsInToken) > 1:
        firstWord = wordsInToken[0].lower()
        secondWord = wordsInToken[1].lower()

        print("\nNew Token")
        for wordIndex in range(0, len(allWords) - 1):
            if firstWord == allWords[wordIndex].lower():
                if secondWord.lower() == allWords[wordIndex + 1].lower():
                    frequency += 1
                    print("freq added: freq - ", frequency)

         # for ind in wordFrequencyDf.index:
         #    if firstWord.lower() == wordFrequencyDf['word'][ind].lower():
         #        frequency += wordFrequencyDf['frequency'][ind]
         #    if secondWord.lower() == wordFrequencyDf['word'][ind].lower():
         #        frequency += wordFrequencyDf['frequency'][ind]


    # if the token is a single word
    else:

        if token in allWordsDict.keys():
            frequency += allWordsDict[token]
            print("freq added: freq - ", frequency)
            #print("\nFrequency added...allWords")

        # if token in wordFrequencyDict.keys():
        #     frequency += wordFrequencyDict[token]
        #     print("\nFrequency added...BNC")


        # for index, row in wordFrequencyDf.iterrows():
        #     #print("\nChecking")
        #     if token == row['word']:
        #         frequency += row['frequency']
        #         print("\nFrequency added..")
        #         globalCount += 1
        #         print("Number: ", globalCount)
        #         break;

    return frequency





def calculateFeatures(cleanDataframe, wordsList, wordFrequencyDict):
    """
        A function to calculate frequency, syllable count and length of the tokens.

        :param cleanDataframe: dataframe of single words and multi-word expressions datasets
        :param wordsList: list of words present in the dataset
        :param wordFrequencyDict: BNC word frequency list

        :return: a dataframe with features of each token
    """

    tokenList = []
    tokenFeatures = pd.DataFrame(columns=['token', 'wordLength', 'syllableCount', 'frequency', 'complexity'])

    for ind in cleanDataframe.index:

        if cleanDataframe['token'][ind] not in tokenList:
            tokenList.append(cleanDataframe['token'][ind].lower())

            length = lengthCount(cleanDataframe['token'][ind])
            syllableCount = getSyllableCount(cleanDataframe['token'][ind])
            frequency = tokenOccurence(cleanDataframe['token'][ind], wordsList, wordFrequencyDict)


            tokenFeatures = tokenFeatures.append({
                                     'token': cleanDataframe['token'][ind].lower(), 'wordLength': length,
                                     'syllableCount': syllableCount, 'frequency': frequency,
                                     'complexity': cleanDataframe['complexity'][ind]},
                                     ignore_index=True)

    return tokenFeatures



def calculateFeatures_test(cleanDataframe, wordsList, wordFrequencyDict):
    """
        A function to calculate frequency, syllable count and length of the tokens.

        :param cleanDataframe: dataframe of single words and multi-word expressions datasets
        :param wordsList: list of words present in the dataset
        :param wordFrequencyDict: BNC word frequency list

        :return: a dataframe with features of each token
    """

    tokenList = []
    tokenFeatures = pd.DataFrame(columns=['token', 'wordLength', 'syllableCount', 'frequency'])

    for ind in cleanDataframe.index:

        if cleanDataframe['token'][ind] not in tokenList:
            tokenList.append(cleanDataframe['token'][ind].lower())

            length = lengthCount(cleanDataframe['token'][ind])
            syllableCount = getSyllableCount(cleanDataframe['token'][ind])
            frequency = tokenOccurence(cleanDataframe['token'][ind], wordsList, wordFrequencyDict)


            tokenFeatures = tokenFeatures.append({
                                     'token': cleanDataframe['token'][ind].lower(), 'wordLength': length,
                                     'syllableCount': syllableCount, 'frequency': frequency},
                                     ignore_index=True)

    return tokenFeatures



def plotFeatures(dataframe):
    """
        A function to plot various features with the target attribute.

        :param dataframe: a dataframe of features
        :return: None
    """

    dataframe.plot(x='frequency', y='complexity', style='o')
    plt.title('Freq of Token in Reference Corpus vs Complexity')
    plt.xlabel('Token Frequency')
    plt.ylabel('Complexity Value')
    plt.show()

    dataframe.plot(x='syllableCount', y='complexity', style='o')
    plt.title('Number of Syllables vs Complexity')
    plt.xlabel('Syllables in the token')
    plt.ylabel('Complexity Value')
    plt.show()

    dataframe.plot(x='wordLength', y='complexity', style='o')
    plt.title('Length of Token vs Complexity')
    plt.xlabel('Length of Token')
    plt.ylabel('Complexity Value')
    plt.show()

    sns.boxplot(y='complexity', x='frequency', data=dataframe)
    plt.show()

    sns.boxplot(y='complexity', x='wordLength', data=dataframe)
    plt.show()

    sns.boxplot(y='complexity', x='syllableCount', data=dataframe)
    plt.show()



def plotHistogram(dataframe):
    """
        A function to plot a histogram for the distribution of the target attribute
        with the syllable count feature and the word length count feature.

        :param dataframe: a dataframe of features
        :return: None
    """

    maximum = max(dataframe['syllableCount'])

    #Histogram for number of syllables
    for count in range(1, maximum+1):
        histDf = pd.DataFrame(columns=['syllableCount', 'complexity'])
        for index in dataframe.index:
            if dataframe['syllableCount'][index] == count:
                histDf = histDf.append({'syllableCount': dataframe['syllableCount'][index],
                                        'complexity': dataframe['complexity'][index]},
                                        ignore_index=True)

        label = "Syllables = " + str(count)
        plt.hist(histDf['complexity'], alpha=0.5, label= label)
        count +=1

    plt.legend(loc='upper right')
    plt.show()


    # Histogram for word length
    maximum = max(dataframe['wordLength'])

    for count in range(1, maximum + 1):
        histDf = pd.DataFrame(columns=['wordLength', 'complexity'])
        for index in dataframe.index:
            if dataframe['wordLength'][index] == count:
                histDf = histDf.append({'wordLength': dataframe['wordLength'][index],
                                        'complexity': dataframe['complexity'][index]},
                                       ignore_index=True)

        label = "Word Length = " + str(count)
        plt.hist(histDf['complexity'], alpha=0.5, label=label)
        count += 1

    plt.legend(loc='upper right')
    plt.show()




def zeroFreqTokens(dataframe):
    """
        A function to plot various features with which have 0 frequency

        :param dataframe: a dataframe of features
        :return: dataframe with all features and frequency = 0
    """

    zeroFreqDf = pd.DataFrame(columns=['token', 'wordLength', 'syllableCount', 'frequency', 'complexity'])

    for ind in dataframe.index:
        if dataframe['frequency'][ind] == 0:
            zeroFreqDf = zeroFreqDf.append({
                'token': dataframe['token'][ind].lower(), 'wordLength': dataframe['wordLength'][ind],
                'syllableCount': dataframe['syllableCount'][ind], 'frequency': dataframe['frequency'][ind],
                'complexity': dataframe['complexity'][ind]},
                ignore_index=True)

    return zeroFreqDf



def zeroFreqTokens_test(dataframe):
    """
        A function to plot various features with which have 0 frequency

        :param dataframe: a dataframe of features
        :return: dataframe with all features and frequency = 0
    """

    zeroFreqDf = pd.DataFrame(columns=['token', 'wordLength', 'syllableCount', 'frequency', 'complexity'])

    for ind in dataframe.index:
        if dataframe['frequency'][ind] == 0:
            zeroFreqDf = zeroFreqDf.append({
                'token': dataframe['token'][ind].lower(), 'wordLength': dataframe['wordLength'][ind],
                'syllableCount': dataframe['syllableCount'][ind], 'frequency': dataframe['frequency'][ind]},
                ignore_index=True)

    return zeroFreqDf




def nonZeroFreqTokens(dataframe):
    """
        A function to plot various features with which have 0 frequency

        :param dataframe: a dataframe of features
        :return: dataframe with all features and frequency = 0
    """

    nonZeroFreqDf = pd.DataFrame(columns=['token', 'wordLength', 'syllableCount', 'frequency', 'complexity'])

    for ind in dataframe.index:
        if dataframe['frequency'][ind] != 0:
            nonZeroFreqDf = nonZeroFreqDf.append({
                'token': dataframe['token'][ind].lower(), 'wordLength': dataframe['wordLength'][ind],
                'syllableCount': dataframe['syllableCount'][ind], 'frequency': dataframe['frequency'][ind],
                'complexity': dataframe['complexity'][ind]},
                ignore_index=True)

    return nonZeroFreqDf


def nonZeroFreqTokens_test(dataframe):
    """
        A function to plot various features with which have 0 frequency

        :param dataframe: a dataframe of features
        :return: dataframe with all features and frequency = 0
    """

    nonZeroFreqDf = pd.DataFrame(columns=['token', 'wordLength', 'syllableCount', 'frequency', 'complexity'])

    for ind in dataframe.index:
        if dataframe['frequency'][ind] != 0:
            nonZeroFreqDf = nonZeroFreqDf.append({
                'token': dataframe['token'][ind].lower(), 'wordLength': dataframe['wordLength'][ind],
                'syllableCount': dataframe['syllableCount'][ind], 'frequency': dataframe['frequency'][ind]},
                ignore_index=True)

    return nonZeroFreqDf


def linearRegression(dataframe):
    """
        The baseline system in order to predict the complexity
        of tokens using the calculated features and evaluate the results.

        :param dataframe: a dataframe of token features
        :return: none
    """

    shape = dataframe.shape
   # print(shape)
   # print(shape[1] - 1)
   # print(shape[1])
    X = dataframe.iloc[:, 1:(shape[1] - 1)].values  # predictors
    y = dataframe.iloc[:, (shape[1] - 1)].values  # target (Complexity)


    ## Splitting into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


    ## Perform linear regression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # print("\nRegressor Intercept: ", regressor.intercept_)
    # print("Regressor Co-efficient: ", regressor.coef_)

    ## Predict complexity value
    y_pred = regressor.predict(X_test)

    # Prediction on training data
    #y_pred = regressor.predict(X_train)

    # Saving Results
    #linearRegPredDf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    #linearRegPredDf.to_csv("LinearReg_nGram_prediction.csv", encoding='utf-8', index=False)

    # function call for predictions and evaluation
    predictionAndEvaluation(y_pred, y_test)

    # function call for train data evaluation
    #predictionAndEvaluation(y_pred, y_train)



def decisionTreeRegression(dataframe):
    """
         A machine learning architecture in order to predict the complexity
         of tokens using the calculated features and evaluate the results.

         :param dataframe: a dataframe of token features
         :return: none
    """
    shape = dataframe.shape
    #print(shape)
    #print(shape[1] - 1)
    #print(shape[1])

    X = dataframe.iloc[:, 1:(shape[1]-1)].values  # predictors
    y = dataframe.iloc[:, (shape[1]-1)].values  # target (Complexity)

    #Splitting into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    regressor = DecisionTreeRegressor(max_depth=6)
    regressor.fit(X_train, y_train)

    ## Predict complexity value
    y_pred = regressor.predict(X_test)

    # Prediction on training data
    #y_pred = regressor.predict(X_train)

    # function call for predictions and evaluation
    predictionAndEvaluation(y_pred, y_test)

    # function call for train data evaluation
    #predictionAndEvaluation(y_pred, y_train)





def MLPRegression(dataframe):
    """
         A machine learning architecture - a neural model in order to predict the complexity
         of tokens using the calculated features and evaluate the results.

         :param dataframe: a dataframe of token features
         :return: none
    """
    shape = dataframe.shape
   # print(shape)
   # print(shape[1] - 1)
   # print(shape[1])

    X = dataframe.iloc[:, 1:(shape[1] - 1)].values  # predictors
    y = dataframe.iloc[:, -1].values  # target (Complexity)

    # Splitting into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

    # Scaling the predictors
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)

    regressor = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="relu", random_state=1, max_iter=2000)
    regressor.fit(X_trainscaled, y_train)

    ## Predict complexity value
    y_pred = regressor.predict(X_test)

    # Prediction on training data
    #y_pred = regressor.predict(X_trainscaled)

    # function call for predictions and evaluation
    predictionAndEvaluation(y_pred, y_test)

    # function call for train data evaluation
    #predictionAndEvaluation(y_pred, y_train)





def randomForestRegression(dataframe):
    """
         A machine learning architecture - a neural model in order to predict the complexity
         of tokens using the calculated features and evaluate the results.

         :param dataframe: a dataframe of token features
         :return: none
    """
    shape = dataframe.shape
   # print(shape)
   # print(shape[1] - 1)
   # print(shape[1])

    X = dataframe.iloc[:, 1:(shape[1] - 1)].values  # predictors
    y = dataframe.iloc[:, -1].values  # target (Complexity)

    # Splitting into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)

    # Scaling the predictors
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    regressor = RandomForestRegressor(n_estimators=50, random_state=0)
    regressor.fit(X_train, y_train)

    ## Predict complexity value
    y_pred = regressor.predict(X_test)

    # Prediction on training data
    #y_pred = regressor.predict(X_trainscaled)

    # function call for predictions and evaluation
    predictionAndEvaluation(y_pred, y_test)

    # function call for train data evaluation
    #predictionAndEvaluation(y_pred, y_train)


def linearRegression_test(dataframe_train, dataframe_test):
    """
        The baseline system in order to predict the complexity
        of tokens using the calculated features and evaluate the results.

        :param dataframe: a dataframe of token features
        :return: none
    """

    shape_train = dataframe_train.shape
    shape_test = dataframe_test.shape

    X_train = dataframe_train.iloc[:, 1:(shape_train[1] - 1)].values  # predictors for train data
    y_train = dataframe_train.iloc[:, (shape_train[1] - 1)].values  # target (Complexity)

    X_test = dataframe_test.iloc[:, 1:shape_test[1]].values  # predictors for test data

    ## Perform linear regression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    ## Predict complexity value
    y_pred = regressor.predict(X_test)

    print("## PREDICTIONS FOR TEST DATA ##\n", y_pred)

    predictionDf = pd.DataFrame({'Predictions': y_pred})
    predictionDf.to_csv('singleWords_linearRegression_predictions.csv', encoding='utf-8', index=False)

    # function call for predictions and evaluation
    # predictionAndEvaluation(y_pred, y_test)



def decisionTreeRegression_test(dataframe_train, dataframe_test):
    """
         A machine learning architecture in order to predict the complexity
         of tokens using the calculated features and evaluate the results.

         :param dataframe: a dataframe of token features
         :return: none
    """
    shape_train = dataframe_train.shape
    shape_test = dataframe_test.shape

    X_train = dataframe_train.iloc[:, 1:(shape_train[1] - 1)].values  # predictors for train data
    y_train = dataframe_train.iloc[:, (shape_train[1] - 1)].values  # target (Complexity)

    X_test = dataframe_test.iloc[:, 1:shape_test[1]].values  # predictors for test data

    regressor = DecisionTreeRegressor(max_depth=6)
    regressor.fit(X_train, y_train)

    ## Predict complexity value
    y_pred = regressor.predict(X_test)

    print("## PREDICTIONS FOR TEST DATA ##\n", y_pred)

    predictionDf = pd.DataFrame({'Predictions': y_pred})
    predictionDf.to_csv('singleWords_decisionTreeRegression_predictions.csv', encoding='utf-8', index=False)

    # function call for predictions and evaluation
    #predictionAndEvaluation(y_pred, y_test)





def MLPRegression_test(dataframe_train, dataframe_test):
    """
         A machine learning architecture - a neural model in order to predict the complexity
         of tokens using the calculated features and evaluate the results.

         :param dataframe: a dataframe of token features
         :return: none
    """
    shape_train = dataframe_train.shape
    shape_test = dataframe_test.shape

    X_train = dataframe_train.iloc[:, 1:(shape_train[1] - 1)].values  # predictors for train data
    y_train = dataframe_train.iloc[:, (shape_train[1] - 1)].values  # target (Complexity)

    X_test = dataframe_test.iloc[:, 1:shape_test[1]].values  # predictors for test data

    # Scaling the predictors
    sc_X = StandardScaler()
    X_trainscaled = sc_X.fit_transform(X_train)
    X_testscaled = sc_X.transform(X_test)

    regressor = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation="relu", random_state=1, max_iter=2000)
    regressor.fit(X_trainscaled, y_train)

    ## Predict complexity value
    y_pred = regressor.predict(X_testscaled)

    print("## PREDICTIONS FOR TEST DATA ##\n", y_pred)

    predictionDf = pd.DataFrame({'Predictions': y_pred})
    predictionDf.to_csv('singleWords_MLPRegression_predictions.csv', encoding='utf-8', index=False)

    # function call for predictions and evaluation
    # predictionAndEvaluation(y_pred, y_test)


def randomForestRegression_test(dataframe_train, dataframe_test):
    """
        The baseline system in order to predict the complexity
        of tokens using the calculated features and evaluate the results.

        :param dataframe: a dataframe of token features
        :return: none
    """

    shape_train = dataframe_train.shape
    shape_test = dataframe_test.shape

    X_train = dataframe_train.iloc[:, 1:(shape_train[1] - 1)].values  # predictors for train data
    y_train = dataframe_train.iloc[:, (shape_train[1] - 1)].values  # target (Complexity)

    X_test = dataframe_test.iloc[:, 1:shape_test[1]].values  # predictors for test data

    ## Perform linear regression
    regressor = RandomForestRegressor(n_estimators=50, random_state=0)
    regressor.fit(X_train, y_train)

    ## Predict complexity value
    y_pred = regressor.predict(X_test)

    print("## PREDICTIONS FOR TEST DATA ##\n", y_pred)

    predictionDf = pd.DataFrame({'Predictions': y_pred})
    predictionDf.to_csv('singleWords_randomForest_predictions.csv', encoding='utf-8', index=False)

    # function call for predictions and evaluation
    # predictionAndEvaluation(y_pred, y_test)




def linearRegressionOnWord2VecModel(dataframe):
    """
        Creating a Word embedding model (Word2Vec) on the tokens in Complex corpus
        and then performing linear regression on 100 dimensions to generate results

        :param dataframe: a dataframe of token features
        :return: none
    """
    all_words = []
    word2VecDict = {}

    for ind in dataframe.index:
        all_words.append(nltk.word_tokenize(dataframe['token'][ind]))

    word2vec = Word2Vec(all_words,size=300, min_count=1)

    for ind in dataframe.index:
        v1 = word2vec.wv[dataframe['token'][ind]]
        word2VecDict.update({dataframe['token'][ind] : v1})


    #creating a dataframe from the dictionary with 300 dimensions
    #the target variable
    vectorDf = pd.DataFrame(data=word2VecDict.values())

    vectorDf['complexity'] = pd.Series(dataframe['complexity'])
    #print(vectorDf)

    print("\nWord2Vec model complete. Executing Linear Regression on the Word2Vec model...")

    #Saving Results
    #vectorDf.to_csv("singleWord_Word2Vec.csv", encoding='utf-8', index = False)

    ## Performing Linear Regression over the vectors ##
    X = vectorDf.iloc[:, 0:300].values # predictors
    y = vectorDf.iloc[:, 300].values  # target (Complexity)

    ## Splitting into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    ## Perform linear regression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)


    ## Predict complexity value
    y_pred = regressor.predict(X_test)

    # Prediction on training data
    #y_pred = regressor.predict(X_train)

    # function call for predictions and evaluation
    predictionAndEvaluation(y_pred, y_test)

    # function call for train data evaluation
    #predictionAndEvaluation(y_pred, y_train)




def characterNgramModel(dataframe):
    """
        An n-gram model at character level. This model aims to understand
        character combinations and their rarity

        :param dataframe: a dataframe of token features
        :return: none
    """
    
    # print(dataframe)
    n = 3
    nGramsDict ={}
    characterFreqDict = {}

    # creating n-grams from each token
    for ind in dataframe.index:
        word = dataframe['token'][ind].lower()
        nGrams = [word[i:i+n] for i in range(len(word)-1)]
        nGramsDict.update({word : nGrams})

    # calculating frequency of each n-gram from all tokens.
    for value in nGramsDict.values():
        for nGram in value:
            if nGram not in characterFreqDict.keys():
                characterFreqDict[nGram] = 1
                dataframe.insert(4, nGram, 0)
            else:
                characterFreqDict[nGram] += 1

    for index in dataframe.index:
        word = dataframe['token'][index].lower()
        nGrams = [word[i:i + n] for i in range(len(word) - 1)]

        for nGram in nGrams:
            dataframe.iloc[index, dataframe.columns.get_loc(nGram)] = 1


    #Arranging the n-Gram frequencies in a descending order to understand rarity
    nGramDescOrderList = sorted(characterFreqDict, key=characterFreqDict.__getitem__, reverse=True)
    nGramDescOrderValueList = sorted(characterFreqDict.values(), reverse= True)
    nGramDf = pd.DataFrame({'n-Grams': nGramDescOrderList, 'Value': nGramDescOrderValueList})

    #Saving Results
    #nGramDf.to_csv("nGramDesc.csv", encoding='utf-8', index=False)
    #print("\n   ### N-GRAM ORDER ###\n", nGramDf )
    #dataframe.to_csv("nGram_attribute.csv", encoding='utf-8', index=False)
    #print("\n\n", dataframe)

    return dataframe



def predictionAndEvaluation(y_pred,y_test):
    """
        A function that generates predictions and evaluation - MAE, MSE, RMSE and
        Kendall Tau distance for all regression models.

        :param y_pred: predicted target values
        :param y_test: actual target values
        :return:
    """

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    print("\n   ### PREDICTIONS ###\n", df)
    
    ## EVALUATION ##
    print("\n### EVALUATION ###")
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 3))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 3))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 3))

    kTau, pValue = stats.kendalltau(y_pred, y_test)
    print("Kendall Tau Distance: ", round(kTau, 3))

    pearsonCorr = df.corr(method="pearson");
    print("Pearson correlation coefficient:\n", round(pearsonCorr,3),"\n\n");



def main():

    # print("Started Execution.\nCleaning Datasets...")
    # singleWords_clean = cleanDataset(singleWords)
    # multiWords_clean = cleanDataset(multiWords)

    # print("Started Execution.\nCleaning Test Datasets...")
    # # singleWords_clean_test = cleanDataset_test(singleWords_test)
    # multiWords_clean_test = cleanDataset_test(multiWords_test)
    #
    # print("Datasets are clean.\nGetting raw text from large Reference Corpora...")
    # rawTextList = getCorpusText()
    # cleanText = cleanCorpusText(rawTextList)
    # wordFrequencyDict = getBNCCorpusFreqList()
    #
    # corpusWords = cleanText.split(" ")
    # corpusWords = pd.Series(corpusWords)
    #
    # countofCorpusWords = len(corpusWords)

    # print("Words counted.\nCalculating features for single words...")
    # singleWords_features = calculateFeatures(singleWords_clean, corpusWords, wordFrequencyDict)
    # print(singleWords_features)
    # singleWords_features.to_csv("singleWord_features.csv", encoding='utf-8', index = False)

    singleWords_features = pd.read_csv("v9_psycho_singleWord_features.csv")


    # print("Words counted.\nCalculating features for single words test dataset...")
    # singleWords_features_test = calculateFeatures_test(singleWords_clean_test, corpusWords, wordFrequencyDict)
    # print(singleWords_features)
    # singleWords_features_test.to_csv("singleWord_features_test.csv", encoding='utf-8', index=False)

    singleWords_features_test = pd.read_csv("singleWord_features_test.csv")



    # print("\nCalculating features for Multi-Word Expressions...")
    # multiWords_features = calculateFeatures(multiWords_clean, corpusWords, wordFrequencyDict)
    # print(multiWords_features)
    # multiWords_features.to_csv("multiWord_features.csv", encoding='utf-8', index=False)

    # multiWords_features = pd.read_csv("multiWord_features.csv")
    #
    #
    # print("\nCalculating features for Multi-Word Expressions Test Dataset...")
    # multiWords_features_test = calculateFeatures_test(multiWords_clean_test, corpusWords, wordFrequencyDict)
    # print(multiWords_features_test)
    # multiWords_features_test.to_csv("multiWord_features_test.csv", encoding='utf-8', index=False)

    # multiWords_features_test = pd.read_csv("multiWord_features_test.csv")

    #print("\nFeatures calculated for both the datasets. Plotting features with the target attribute...")
    #nonZeroFreqDf = nonZeroFreqTokens(singleWords_features)
    # plotFeatures(nonZeroFreqDf)
    # plotHistogram(singleWords_features)



    # mweNonZeroFreqDf = nonZeroFreqTokens(multiWords_features)
    # plotFeatures(mweNonZeroFreqDf)
    # plotHistogram(multiWords_features)

    #print("\nFinding Tokens with 0 Frequency and Plotting for trends...")
    #zeroFreqDf = zeroFreqTokens(singleWords_features)
    #plotFeatures(zeroFreqDf)
    #plotHistogram(zeroFreqDf)
    #plotFeatures(multiWords_features)



    ### LINEAR REGRESSION ###
    print("\nFeatures plotted. Executing Linear regression for Single word Tokens...")
    # linearRegression(singleWords_features)

    # print("\nLinear Regression for Multi-word Expressions...")
    # linearRegression(multiWords_features)

    print("\nFeatures plotted. Executing Linear regression for Single word Tokens Test Dataset...")
    # linearRegression_test(singleWords_features, singleWords_features_test)

    # print("\nLinear Regression for Multi-word Expressions Test Dataset...")
    # linearRegression_test(multiWords_features, multiWords_features_test)



    ### DECISION TREE REGRESSION ###
    print("\nLinear Regression Complete. Executing Decision Tree regression for Single word Tokens...")
    # decisionTreeRegression(singleWords_features)

    # print("\nDecision Tree Regression for Multi-word Expressions...")
    # decisionTreeRegression(multiWords_features)

    print("\nLinear Regression Complete. Executing Decision Tree regression for Single word Tokens Test Dataset...")
    # decisionTreeRegression_test(singleWords_features, singleWords_features_test)

    # print("\nDecision Tree Regression for Multi-word Expressions Test Dataset...")
    # decisionTreeRegression_test(multiWords_features, multiWords_features_test)



    ### MULTI LAYER PERCEPTRON REGRESSION ###
    print("\nDecision Tree Regression Complete. Executing Multi Layer Perceptron (Neural Model) regression for Single word Tokens...")
    # MLPRegression(singleWords_features)

    # print("\nMulti Layer Perceptron Regression for Multi-word Expressions...")
    # MLPRegression(multiWords_features)

    print("\nDecision Tree Regression Complete. Executing Multi Layer Perceptron (Neural Model) regression "
          "for Single word Tokens Test Dataset...")
    # MLPRegression_test(singleWords_features, singleWords_features_test)

    # print("\nDecision Tree Regression Complete. Executing Multi Layer Perceptron (Neural Model) regression "
    #       "for Multi Word Expressions Test Dataset...")
    # MLPRegression_test(multiWords_features, multiWords_features_test)

    ### RANDOM FOREST REGRESSION ###
    print("\nMLP Regression Complete. Executing Random Forest regression for Single word Tokens...")
    # randomForestRegression(singleWords_features)

    # print("\nMulti Layer Perceptron Regression for Multi-word Expressions...")
    # randomForestRegression(multiWords_features)

    print("\nMLP Regression Complete. Executing Random Forest regression "
          "for Single word Tokens Test Dataset...")
    # randomForestRegression_test(singleWords_features, singleWords_features_test)

    # print("\nMLP Regression Complete. Executing Random Forest regression "
    #       "for Multi Word Expressions Test Dataset...")
    # randomForestRegression_test(multiWords_features, multiWords_features_test)


    ### WORD EMBEDDING MODEL ###
    # print("\nMLP Regression Complete. Creating a Word2Vec model for Single word Tokens...")
    #linearRegressionOnWord2VecModel(singleWords_features)


    ### CHARACTER N-GRAM MODEL ###
    nGramAtrributeDf = characterNgramModel(singleWords_features)
    
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(nGramAtrributeDf)
    
    ### RANDOM FOREST REGRESSION with CHARACTER N-GRAMs ###    
    print("\nMLP Random Forest regression Complete. Executing Random Forest regression with Character N-grams "
      "for Single word Tokens Test Dataset...")
    
    
    
    randomForestRegression(nGramAtrributeDf)
    
    # ### REGRESSION ANALYSIS ON CHARACTER GRAM MODEL ###
    # print("\nExecuting Regression analysis on Character nGram Model. Executing Linear Regression... ")
    # linearRegression(nGramAtrributeDf)
    # print("\nLinear Regression Complete. Executing Decision Tree regression for Single word Tokens...")
    # decisionTreeRegression(nGramAtrributeDf)
    # print("\nDecision Tree Regression Complete. Executing MLP regression for Single word Tokens...")
    # MLPRegression(nGramAtrributeDf)

if __name__ == '__main__':
    main()