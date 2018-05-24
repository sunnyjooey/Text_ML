import pandas as pd
import operator

def get_svo_sentiments(df, lex, label_col, window, count_thresh):
    """
    Given a dataframe, returns sentiments dictionary of words and 
    their associated sentiment score (ranging -1 ~ 1)
    Return dictionary is disaggregated by subject/object and label: 
        sentiments = {'object': {'d': {'word': float ...}}}
    
    Requires `lex`, a sentiment lexicon (dictionary of word key, sentiment val)
    `window` refers to number of words to check above and below target word
    Return dictionary discludes words appearing less than `count_thresh` times
    """
    # get sentiment around svo (aggregate and count of tweets)
    svo_sentiments = {
        'subject': {},
        'object': {}
    }
    
    labels = df[label_col].unique()
    for label in labels:
        for key in svo_sentiments.keys():
            svo_sentiments[key][label] = {}
       
        # all svos for one label
        label_svos = df[df[label_col] == label].reset_index()
        for i in range(label_svos.shape[0]):
            user_svo_list = label_svos.loc[i,'svos']
            for tweet_svo_list in user_svo_list:
                for svo in tweet_svo_list:
                    s = svo[0]
                    o = svo[2]
                    # checking s and o in context
                    tokenized_text_lists = label_svos.loc[i,'tokenized_text_agg']
                    for tokenized_text in tokenized_text_lists:
                        
                        # for 's'ubjects
                        if s != '' and '/' not in s:
                            if s in tokenized_text:
                                s_index = tokenized_text.index(s)
                               
                                # get words in window
                                lower_bound = max(0, s_index-window)
                                upper_bound = min(len(tokenized_text), s_index+1+window)
                                pre_words = tokenized_text[lower_bound : s_index]
                                post_words = tokenized_text[s_index+1: upper_bound]
                                window_words = pre_words + post_words
                                
                                # tally up sentiment score for s 
                                sentiment_total = 0
                                sentiment_count = 0
                                for word in window_words:
                                    if word in lex:
                                        sentiment_total += lex[word]
                                        sentiment_count += 1
                                
                                # build dictionary
                                if sentiment_count != 0:
                                    avg_sentiment = sentiment_total/sentiment_count
                                    if s in svo_sentiments['subject'][label]:
                                        svo_sentiments['subject'][label][s]['total_sentim'] += avg_sentiment
                                        svo_sentiments['subject'][label][s]['count'] += 1    
                                    else:
                                        svo_sentiments['subject'][label][s] = {'total_sentim': avg_sentiment, 'count': 1}
                                else:
                                    continue
                        
                        # for 'o'bjects
                        if o != '' and '/' not in o:
                            if o in tokenized_text:
                                o_index = tokenized_text.index(o)
                               
                                # get words in window
                                lower_bound = max(0, o_index-window)
                                upper_bound = min(len(tokenized_text), o_index+1+window)
                                pre_words = tokenized_text[lower_bound : o_index]
                                post_words = tokenized_text[o_index+1: upper_bound]
                                window_words = pre_words + post_words
                                
                                # tally up score for s 
                                sentiment_total = 0
                                sentiment_count = 0
                                for word in window_words:
                                    if word in lex:
                                        sentiment_total += lex[word]
                                        sentiment_count += 1
                                
                                if sentiment_count != 0:
                                    avg_sentiment = sentiment_total/sentiment_count
                                    if o in svo_sentiments['object'][label]:
                                        svo_sentiments['object'][label][o]['total_sentim'] += avg_sentiment
                                        svo_sentiments['object'][label][o]['count'] += 1    
                                    else:
                                        svo_sentiments['object'][label][o] = {'total_sentim': avg_sentiment, 'count': 1}
                                else:
                                    continue
    
    # get average sentiment (if count above threshold) per label, per word 
    sentiments = {
        'subject': {},
        'object': {}
    }
    for label in labels:
        for key in sentiments.keys():
            sentiments[key][label] = {}
            
    for sv in svo_sentiments.keys():
        for label_key in svo_sentiments[sv]:
            for word_key in svo_sentiments[sv][label_key]:
                if svo_sentiments[sv][label_key][word_key]['count'] >= count_thresh:
                    sentiments[sv][label_key][word_key] = svo_sentiments[sv][label_key][word_key]['total_sentim'] / svo_sentiments[sv][label_key][word_key]['count']

    return sentiments


def most_pos_neg_sents(sent_dict, k):
    """
    Given sentiments dictionary (from `svo_sentiments` function),
    return dictionary of the k most positive subjects, k most positive objects,
    k most negative subjects, k most negative objects for EACH label 
    (currently only works for the 2 label case). Word that are pos/neg in both labels
    will be discluded.
    """
    k_sentiment_dict = {}
    # for subject/object
    for sv in sent_dict.keys():
        # for each label
        for label in sent_dict[sv].keys():
            if label not in k_sentiment_dict:
                k_sentiment_dict[label] = {}
            
            # sort words by sentiment value
            sorted_list = sorted(sent_dict[sv][label].items(), key=operator.itemgetter(1))
            sorted_list.reverse()
            
            # get k most positive
            kpos_list = sorted_list[:k] 
            kpos_list = [pair[0] for pair in kpos_list]
            if 'POSITIVE' in k_sentiment_dict[label].keys():
                k_sentiment_dict[label]['POSITIVE'] += kpos_list
            else:
                k_sentiment_dict[label]['POSITIVE'] = kpos_list
            
            # get k most negative words
            kneg_list = sorted_list[len(sorted_list)-k :]
            kneg_list = [pair[0] for pair in kneg_list]
            if 'NEGATIVE' in k_sentiment_dict[label]:
                k_sentiment_dict[label]['NEGATIVE'] += kneg_list
            else:
                k_sentiment_dict[label]['NEGATIVE'] = kneg_list
    
    # now find unique and common pos/neg words between labels (only works for 2 labels!)
    labels = list(k_sentiment_dict.keys())
    sentiments = ['POSITIVE', 'NEGATIVE']
    
    separate_sentiment = {}
    for sentiment in sentiments: 
        label1_pos_words = k_sentiment_dict[labels[0]][sentiment]
        label2_pos_words = k_sentiment_dict[labels[1]][sentiment]
        label1_pos_unique = [word for word in label1_pos_words if word not in label2_pos_words]
        label2_pos_unique = [word for word in label2_pos_words if word not in label1_pos_words]
        # words that have the same sentiment in both labels -- to be discluded
        common_pos = [word for word in label1_pos_words if word in label2_pos_words]

        if labels[0] not in separate_sentiment:
            separate_sentiment[labels[0]] = {}
        separate_sentiment[labels[0]][sentiment] = label1_pos_unique
        if labels[1] not in separate_sentiment:
            separate_sentiment[labels[1]] = {}
        separate_sentiment[labels[1]][sentiment] = label2_pos_unique
        separate_sentiment[sentiment + '_COMMON'] = common_pos   
        
    return separate_sentiment


def featurize_sentiments(df, label_col, tok_text_col, lex, window, count_thresh, k, sent_dict=None):
    """
    Featurize most positive/negative subject/object words in each label
    Value is the average sentiment of the `window` number of words before
    and after each sentiment-laden word
    Inputs:
        tok_text_col: list of lists, each inner list is a tokenized tweet
        lex: sentiment lexicon, dictionary of word-key, sentiment-val
        window: window of words to consider
        count_thresh: drop words appearing less than threshold times
        k: top k sentiment laden words for each pos/neg sentiment, sub/obj pos, and label (= 2*2*2*k )
        sent_dict: for the test set only -- pass in sent_dict produced from the training set 
                    for training set, keep as None
    Output: featurized dataframe, sent_dict (if produced from training set, use for testing set)
    """
    if not sent_dict:
        # Get sentiments for all words in text
        all_sent_dict = get_svo_sentiments(df, lex, label_col, window, count_thresh)
        # Training - get most sentiment-laden words
        sent_dict = most_pos_neg_sents(all_sent_dict, k)
    
    # Featurize
    labels = df[label_col].unique()
    pos_neg = ['NEGATIVE', 'POSITIVE']
    
    for label in labels:
        for pn in pos_neg:
            
            # feature words for one label (dem/rep), one sentiment type (positive/negative)
            sent_features = sent_dict[label][pn]
            for i in range(df.shape[0]):
                tokenized_text_lists = df.loc[i,tok_text_col]
                i_sent_dict = {}
                for tokenized_text in tokenized_text_lists:
                    for sent in sent_features:
                        if sent in tokenized_text:
                            sent_index = tokenized_text.index(sent)
                            
                            # get words in window
                            lower_bound = max(0, sent_index-window)
                            upper_bound = min(len(tokenized_text), sent_index+1+window)
                            pre_words = tokenized_text[lower_bound : sent_index]
                            post_words = tokenized_text[sent_index+1 : upper_bound]
                            window_words = pre_words + post_words
                            
                            # tally up score for feature word sentiment
                            sentiment_total = 0
                            sentiment_count = 0
                            for word in window_words:
                                if word in lex:
                                    sentiment_total += lex[word]
                                    sentiment_count += 1
                                    
                            if sentiment_count != 0:
                                # since each word can appear more than once, get avgerage sentiment
                                avg_sentiment = sentiment_total/sentiment_count
                                if 'sent_' + sent in i_sent_dict:
                                    i_sent_dict['sent_' + sent]['avg'] += avg_sentiment
                                    i_sent_dict['sent_' + sent]['count'] +=  1
                                else:
                                    i_sent_dict['sent_' + sent] = {'avg': avg_sentiment, 'count': 1}                   
                
                # create features from each user i's dictionary
                overall_avg = 0
                overall_count = 0
                for sent_key in i_sent_dict.keys():
                    df.loc[i, sent_key] = i_sent_dict[sent_key]['avg'] / i_sent_dict[sent_key]['count']
                    overall_avg += i_sent_dict[sent_key]['avg']
                    overall_count += i_sent_dict[sent_key]['count']
                # create overall sentiment score for label, sentiment type
                if overall_count != 0:
                    df.loc[i, label +'_'+ pn] = overall_avg / overall_count
                
    df.fillna(0, inplace=True)
    
    return df, sent_dict