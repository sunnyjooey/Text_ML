import pandas as pd
from sklearn.model_selection import train_test_split
import proto
import topics
import sentiments
import time


def featurize(df, label, user_id, test_size, proto_word_args=None, hashtag_args=None, topic_model_args=None, topic_model_params=None, sent_args=None, random_state=None):
    """
    Featurize the dataframe
    """
    # insurance
    df = df.drop_duplicates(subset=[user_id], keep='first')

    # train test split
    df_x = df.drop(label, axis=1)
    df_y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=test_size, random_state=random_state)
    # add y back in for feature generation
    X_train[label] = y_train 
    
    # intialize feature dfs 
    X_train_ft = pd.DataFrame()
    X_test_ft = pd.DataFrame()

    # keep track of trained objects
    trained_objects = {}

    orig_cols = list(df.columns)
    orig_cols.remove(user_id)

    time_start = time.time()

    if proto_word_args:
        # create proto word features for train and test
        proto_words = proto.protoWords()
        proto_words.fit(X_train, label, proto_word_args['text_col'], \
                        user_id, proto_word_args['tok_type'], \
                        isalpha=proto_word_args['isalpha'])
        X_train_word_ft = proto.create_protoword_features(X_train, proto_words, \
                                                          proto_word_args['text_col'], user_id, \
                                                          proto_word_args['tok_type'], proto_word_args['isalpha'], \
                                                          proto_word_args['top_k'], proto_word_args['word_count_thresh'])
        X_test_word_ft = proto.create_protoword_features(X_test, proto_words, proto_word_args['text_col'], user_id, \
                                                          proto_word_args['tok_type'], proto_word_args['isalpha'], \
                                                          proto_word_args['top_k'], proto_word_args['word_count_thresh'])
        trained_objects['proto_words'] = proto_words
        
        # create proto hashtag features for train and test
        proto_ht = proto.protoHashtags()
        proto_ht.fit(X_train, label, hashtag_args['text_col'], user_id)
        X_train_ht_ft = proto.create_protohashtag_features(X_train, proto_ht, hashtag_args['text_col'], user_id, \
                                                           hashtag_args['top_k'], hashtag_args['ht_count_thresh'])
        X_test_ht_ft = proto.create_protohashtag_features(X_test, proto_ht, hashtag_args['text_col'], user_id, \
                                                           hashtag_args['top_k'], hashtag_args['ht_count_thresh'])
        trained_objects['proto_ht'] = proto_ht
    
        # becomes feature df
        X_train_ft = pd.merge(X_train_word_ft, X_train_ht_ft, on=user_id)
        X_test_ft = pd.merge(X_test_word_ft, X_test_ht_ft, on=user_id)
        
        t =  time.time()
        print("Took {} seconds to featurize proto words and hashtags".format(str(int(t - time_start))))
        time_start = t

    if topic_model_args:
        # create topic features 
        X_train_proc = topics.process_text(X_train, topic_model_args['text_col'], topic_model_args['stops'], \
                                   topic_model_args['stemmer'], topic_model_args['lemmer'])
        X_test_proc = topics.process_text(X_test, topic_model_args['text_col'], topic_model_args['stops'], \
                                   topic_model_args['stemmer'], topic_model_args['lemmer'])
        TFVectorizer, dictionary, lda, wordRanksDF = topics.topic_model(X_train_proc, topic_model_args['text_col'], user_id,\
                                                                        topic_model_params['num_topic'], topic_model_params['max_df'], \
                                                                        topic_model_params['min_df'], topic_model_params['max_feature'],\
                                                                        topic_model_args['stops'], topic_model_params['alpha'],\
                                                                        topic_model_params['eta'], random_state, topic_model_params['serialized'])    
        X_train_topic_ft = topics.get_topic_features(X_train_proc, user_id, dictionary, TFVectorizer, lda)
        X_test_topic_ft = topics.get_topic_features(X_test_proc, user_id, dictionary, TFVectorizer, lda)
        
        trained_objects['topic_words'] = wordRanksDF
    
        if X_train_ft.shape[0] != 0:
            # merge feature dfs
            X_train_ft = pd.merge(X_train_ft, X_train_topic_ft, on=user_id)
            X_test_ft = pd.merge(X_test_ft, X_test_topic_ft, on=user_id)
        else:
            # becomes feature df
            X_train_ft = X_train_topic_ft
            X_test_ft = X_test_topic_ft
        
        t =  time.time()
        print("Took {} seconds to featurize topics".format(str(int(t - time_start))))
        time_start = t


    if sent_args:
        # create sentiment features
        label_list = df_y.unique()
        sent_dict = sentiments.most_pos_neg_sents(X_train, label, sent_args['lexicon'], sent_args['window'], sent_args['count_thresh'], sent_args['top_k'])
        X_train_sent_ft = sentiments.featurize_sentiments(X_train, label_list, user_id, sent_args['tok_text_col'], sent_dict, sent_args['lexicon'], sent_args['window'])
        X_test_sent_ft = sentiments.featurize_sentiments(X_test, label_list, user_id, sent_args['tok_text_col'], sent_dict, sent_args['lexicon'], sent_args['window'])

        trained_objects['sent_dict'] = sent_dict

        if X_train_ft.shape[0] != 0:
            # merge feature dfs
            X_train_ft = pd.merge(X_train_ft, X_train_sent_ft, on=user_id)
            X_test_ft = pd.merge(X_test_ft, X_test_sent_ft, on=user_id)
        else:
            # becomes feature df
            X_train_ft = X_train_sent_ft
            X_test_ft = X_test_sent_ft
        
        # drop original columns
        #X_train_ft = X_train_ft.drop(orig_cols, axis=1)
        #orig_cols.remove(label)
        #X_test_ft = X_test_ft.drop(orig_cols, axis=1)
        
        t =  time.time()
        print("Took {} seconds to featurize sentiment words".format(str(int(t - time_start))))
        time_start = t
    
    # drop cols
    X_train_ft = X_train_ft.drop(user_id, axis=1)
    X_test_ft = X_test_ft.drop(user_id, axis=1)

    # drop features in training that is missing in testing
    test_missing_cols = [col for col in X_train_ft.columns if col not in X_test_ft.columns]
    X_train_ft = X_train_ft.drop(test_missing_cols, axis=1)

    if 'tokenized_text' in X_train_ft.columns:
    	X_train_ft = X_train_ft.drop(['tokenized_text', 'normalized_tokens', 'reduced_tokens'], axis=1)
    	X_test_ft = X_test_ft.drop(['tokenized_text', 'normalized_tokens', 'reduced_tokens'], axis=1) 
    
    return X_train_ft, X_test_ft, y_train, y_test, trained_objects