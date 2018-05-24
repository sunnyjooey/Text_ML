import pandas as pd
from sklearn.model_selection import train_test_split
import proto
import topics
import sentiments


def featurize(df, label, test_size, proto_word_args=None, hashtag_args=None, topic_model_args=None, topic_model_params=None, sent_args=None, random_state=None):
	"""
	Featurize the dataframe
	"""
    # train test split
    df_x = df.drop(label, axis=1)
    df_y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=test_size, random_state=random_state)
    # add y back in for feature generation
    X_train[label] = y_train 

    # intialize feature dfs 
    X_train_ft = pd.DataFrame()
    X_test_ft = pd.DataFrame()

    # trained objects
    trained_objects = {}

    if proto_words_args:
	    # create proto word features for train and test
	    proto_words = proto.protoWords()
	    proto_words.fit(X_train, label, proto_word_args['text_col'], \
	                    proto_word_args['user_id'], proto_word_args['tok_type'], \
	                    isalpha=proto_word_args['isalpha'])
	    X_train_word_ft = proto.create_protoword_features(X_train, proto_words, \
	                                                      proto_word_args['text_col'], proto_word_args['user_id'], \
	                                                      proto_word_args['tok_type'], proto_word_args['isalpha'], \
	                                                      proto_word_args['top_k'], proto_word_args['word_count_thresh'])
	    X_test_word_ft = proto.create_protoword_features(X_test, proto_words, proto_word_args['text_col'], proto_word_args['user_id'], \
	                                                      proto_word_args['tok_type'], proto_word_args['isalpha'], \
	                                                      proto_word_args['top_k'], proto_word_args['word_count_thresh'])
	    trained_objects['proto_words'] = proto_words
	    
	    # create proto hashtag features for train and test
	    proto_ht = proto.protoHashtags()
	    proto_ht.fit(X_train, label, hashtag_args['text_col'], hashtag_args['user_id'])
	    X_train_ht_ft = proto.create_protohashtag_features(X_train, proto_ht, hashtag_args['text_col'], hashtag_args['user_id'], \
	                                                       hashtag_args['top_k'], hashtag_args['ht_count_thresh'])
	    X_test_ht_ft = proto.create_protohashtag_features(X_test, proto_ht, hashtag_args['text_col'], hashtag_args['user_id'], \
	                                                       hashtag_args['top_k'], hashtag_args['ht_count_thresh'])
	    trained_objects['proto_ht'] = proto_ht

	    # becomes feature df
	    X_train_ft = pd.merge(X_train_word_ft, X_train_ht_ft, on=hashtag_args['user_id'])
	    X_test_ft = pd.merge(X_test_word_ft, X_test_ht_ft, on=hashtag_args['user_id'])
    

    if topic_model_args:
	    # create topic features 
	    X_train_proc = topics.process_text(X_train, topic_model_args['text_col'], topic_model_args['stops'], \
	                               topic_model_args['stemmer'], topic_model_args['lemmer'])
	    X_test_proc = topics.process_text(X_test, topic_model_args['text_col'], topic_model_args['stops'], \
	                               topic_model_args['stemmer'], topic_model_args['lemmer'])
	    TFVectorizer, dictionary, lda, wordRanksDF = topics.topic_model(X_train_proc, topic_model_args['text_col'], topic_model_args['user_id'],\
	                                                                    topic_model_params['num_topic'], topic_model_params['max_df'], \
	                                                                    topic_model_params['min_df'], topic_model_params['max_feature'],\
	                                                                    topic_model_args['stops'], topic_model_params['alpha'],\
	                                                                    topic_model_params['eta'], random_state, topic_model_params['serialized'])    
	    X_train_topic_ft = topics.get_topic_features(X_train_proc, topic_model_args['user_id'], dictionary, TFVectorizer, lda)
	    X_test_topic_ft = topics.get_topic_features(X_test_proc, topic_model_args['user_id'], dictionary, TFVectorizer, lda)
	    
	    trained_objects['topic_words'] = wordRanksDF

	    if X_train_ft.shape[0] != 0:
		    # merge feature dfs
		    X_train_ft = pd.merge(X_train_ft, X_train_topic_ft, on=topic_model_args['user_id'])
		    X_test_ft = pd.merge(X_test_ft, X_test_topic_ft, on=topic_model_args['user_id'])
    	else:
    		# becomes feature df
    		X_train_ft = X_train_topic_ft
    		X_test_ft = X_test_topic_ft


    if sent_args:
    	# create sentiment features
    	
    	sent_dict = sentiments.most_pos_neg_sents(X_train, label, sent_args['lexicon'], sent_args['window'], sent_args['count_thresh'], sent_args['top_k'])

    	X_train_sent_ft = sentiments.featurize_sentiments(X_train, label, sent_args['tok_text_col'], sent_dict, sent_args['lexicon'], sent_args['window'])
    	X_test_sent_ft = sentiments.featurize_sentiments(X_test, label, sent_args['tok_text_col'], sent_dict, sent_args['lexicon'], sent_args['window'])

    	if X_train_ft.shape[0] != 0:
		    # merge feature dfs
		    X_train_ft = pd.merge(X_train_ft, X_train_sent_ft, on=topic_model_args['user_id'])
		    X_test_ft = pd.merge(X_test_ft, X_test_sent_ft, on=topic_model_args['user_id'])
    	else:
    		# becomes feature df
    		X_train_ft = X_train_sent_ft
    		X_test_ft = X_test_sent_ft

    	trained_objects['sent_dict'] = sent_dict

    return X_train_ft, X_test_ft, y_train, y_test, trained_objects