import pandas as pd
import gensim
import nltk 
import sklearn


def process_text(df, text_col, stop_words, stemmer, lemmer):
    """
    process text for topic modeling
    """
    df['tokenized_text'] = df[text_col].apply(lambda x: nltk.word_tokenize(x))
    df['normalized_tokens'] = df['tokenized_text'].apply(lambda x: normalizeTokens(x, stopwordLst=stop_words, stemmer=stemmer, lemmer=lemmer))
    return df


def dropMissing(wordLst, vocab):
    return [w for w in wordLst if w in vocab]


def normalizeTokens(tokenLst, stopwordLst=None, stemmer=None, lemmer=None):
    """
    normalizes the tokenized text
    from: 
        https://github.com/Computational-Content-Analysis-2018/lucem_illud/blob/master/lucem_illud/proccessing.py
    """
    #Lowering the case and removing non-words
    workingIter = (w.lower() for w in tokenLst if w.isalpha())

    #Now we can use the semmer, if provided
    if stemmer is not None:
        workingIter = (stemmer.stem(w) for w in workingIter)

    #And the lemmer
    if lemmer is not None:
        workingIter = (lemmer.lemmatize(w) for w in workingIter)

    #And remove the stopwords
    if stopwordLst is not None:
        workingIter = (w for w in workingIter if w not in stopwordLst)
        
    #We will return a list with the stopwords removed
    return list(workingIter)


def topic_model(df, text_col, user_id, num_topic, max_df, min_df, max_feature, stop_words, alpha, eta, random_number=None, serialized=None):
    """
    Outputs top topics words according throughout all the hyperparameter combinations.
    Params:
        df: Dataframe with text to be analyzed. Requires column named "normalized_tokens" with text 
            as tokens that have been stemmed (and lemmatized).
        text_col: column with the text to be modeled
        random_number: Seed for the LdaModel. Default None will give out a different model every 
            time for the same parameter combination.
        All other parameters are lists of that param name to be passed to gensim.LdaModel
    Modified from: 
        https://github.com/Computational-Content-Analysis-2018/Content-Analysis/tree/master/3-Clustering-and-Topic-Modeling
    """ 
    TFVectorizer = sklearn.feature_extraction.text.TfidfVectorizer(max_df=max_df, min_df=min_df, max_features=max_feature, stop_words=stop_words)
    TFVects = TFVectorizer.fit_transform(df[text_col])

    # drop
    reduced_token_str = 'reduced_tokens' 
    norm_col = 'normalized_tokens'
    df[reduced_token_str] = df[norm_col].apply(lambda x: dropMissing(x, TFVectorizer.vocabulary_.keys()))

    # Make dictionary from reduced tokens
    dictionary = gensim.corpora.Dictionary(df[reduced_token_str])

    if serialized != None:
        corpus = gensim.corpora.MmCorpus(serialized)
    
    else:
        # Make corpus
        corpus = [dictionary.doc2bow(text) for text in df[reduced_token_str]]
        # Serialize
        gensim.corpora.MmCorpus.serialize('serial.mm', corpus)
    
    # Topic model
    lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topic, 
                                          alpha=alpha, eta=eta, random_state=random_number)
    topicsDict = {}
    for topicNum in range(lda.num_topics):
        topicWords = [w for w in lda.show_topic(topicNum, topn=20)]
        topicsDict['Topic_{}'.format(topicNum)] = topicWords
    wordRanksDF = pd.DataFrame(topicsDict)
    return TFVectorizer, dictionary, lda, wordRanksDF


def get_topic_features(df, user_id, dictionary, TFVectorizer, lda):
    """
    Input dictionary, TFVectorizer, lda from outputs of topic_model function
    Outputs ldaDF of topic proportions for each user
    """
    reduced_token_str = 'reduced_tokens'
    norm_col = 'normalized_tokens'
    if reduced_token_str not in df.columns:
        df[reduced_token_str] = df[norm_col].apply(lambda x: dropMissing(x, TFVectorizer.vocabulary_.keys()))
    
    # Topics per user
    ldaDF = pd.DataFrame({
            user_id : df[user_id],
            'topics' : [lda[dictionary.doc2bow(l)] for l in df[reduced_token_str]]})    
    topicsProbDict = {i : [0] * len(ldaDF) for i in range(lda.num_topics)}
    
    for index, topicTuples in enumerate(ldaDF['topics']):
        for topicNum, prob in topicTuples:
            topicsProbDict[topicNum][index] = prob
    for topicNum in range(lda.num_topics):
        ldaDF['topic_{}'.format(topicNum)] = topicsProbDict[topicNum]
    
    ldaDF.drop(['topics'], axis=1, inplace=True)
    return ldaDF