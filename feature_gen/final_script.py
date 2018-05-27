# For running all three feature groups

import pandas as pd
import nltk
import pickle
import operator
import feature

# Import processed data: see data_preprocessing folder for details
with open('./data/svo_df.pkl', 'rb') as f:
    df = pickle.load(f)

# Import sentiment lexicon, from: https://github.com/zeeeyang/lexicon_rnn/blob/master/lexicons/sspe.lex2
ts_lex = {}
with open('./data/ts_lex.txt','r',encoding='utf-8') as f:
    for line in f:
        entry = line.split(' ')
        ts_lex[entry[0]] = float(entry[1])

proto_word_args_1 = {
    'text_col': 'full_text_agg', 
    'tok_type': 'clean', 
    'isalpha': True,
    'top_k': 100,
    'word_count_thresh': 5
}

proto_word_args_2 = {
    'text_col': 'full_text_agg', 
    'tok_type': 'clean', 
    'isalpha': True,
    'top_k': 300,
    'word_count_thresh': 5
}

hashtag_args_1 = {
    'text_col': 'hashtags_agg', 
    'top_k': 50,
    'ht_count_thresh': 3 
}

hashtag_args_2 = {
    'text_col': 'hashtags_agg', 
    'top_k': 100,
    'ht_count_thresh': 3 
}

topic_model_args = {
    'text_col': 'clean_text_agg',
    'stops': nltk.corpus.stopwords.words('english') + ['rt'],
    'stemmer': nltk.stem.snowball.SnowballStemmer('english'), 
    'lemmer': None
}

topic_model_params_1 = {
    'num_topic': 50, 
    'max_df': 0.5, 
    'min_df': 1, 
    'max_feature': 1000, 
    'alpha': 0.1, 
    'eta': 0.1,  
    'serialized': None 
}

topic_model_params_2 = {
    'num_topic': 100, 
    'max_df': 0.5, 
    'min_df': 1, 
    'max_feature': 1000, 
    'alpha': 0.1, 
    'eta': 0.1,  
    'serialized': None 
}

sent_args_1 = {
    'lexicon': ts_lex,
    'window': 4,
    'count_thresh': 6,
    'top_k': 20,
    'tok_text_col': 'tokenized_text_agg'
    }

sent_args_2 = {
    'lexicon': ts_lex,
    'window': 4,
    'count_thresh': 6,
    'top_k': 50,
    'tok_text_col': 'tokenized_text_agg'
    }


X_train_ft, X_test_ft, y_train, y_test, obj = feature.featurize(df, 'label', 'user_id', 0.3, proto_word_args=proto_word_args_2, hashtag_args=hashtag_args_2, topic_model_args=topic_model_args, topic_model_params=topic_model_params_1, sent_args=sent_args_1)

X_train_ft.to_pickle('./all_features/X_train_ft.pkl')
X_test_ft.to_pickle('./all_features/X_test_ft.pkl')
y_train.to_pickle('./all_features/y_train.pkl')
y_test.to_pickle('./all_features/y_test.pkl')
pickle.dump(obj, open('./all_features/obj.pkl', 'wb'))

