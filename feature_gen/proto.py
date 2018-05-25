import pandas as pd
import operator
import nltk 

class protoWords():
    """
    Get the k most prototypical words for each class (label)
    """
    def __init__(self):
        self._word_count_dict = {'total': {}}
        self._score_dict = {}
        self._k_dict = {}
        
    def fit(self, df, label_col, text_col, user_id, tok_type, isalpha=True):
        self._get_word_count(df, label_col, text_col, user_id, tok_type, isalpha)
        self._get_scores()
        
    def top_k(self, k, thresh):
        self._get_top_k(k, thresh)
        return self._k_dict
    
    def _get_word_count(self, df, label_col, text_col, user_id, tok_type, isalpha):
        """
        inputs:
            df: note that text must already be aggregated by user 
            label_col: label (class) column
            text_col: text column, text is one long string
            user_id: user's id column
            tok_type: 'full' (use lower case text) 
                    or 'clean' (the default, use lower case text, no handles, reduced word length)
            isalpha: True (only count words with letters) or False (also count words with numbers, punctuation, etc.)
        
        builds:
            count of words in each label:
            self._word_count_dict = {
            'label_1': {
                'word_1': 23,
                'word_2': 12, ...
                }, ...
            }
            
          #  count of total words used by each user:
           # self._user_word_count ={
            #user_1_id: 231,
            #user_2_id: 123, ...
            #}
        """
        # twitter tokenizer default: user lower case
        if tok_type == 'full':
            tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False)
        # (default) twitter tokenizer super clean: use lower case, no handles, reduce exaggerated word length
        # note that hashtags remain (#icecream and icecream count separately), urls remain    
        else:
            tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

        # labels (classes)
        labels = df[label_col].unique()
        for label in labels:
            self._word_count_dict[label] = {}
            
            # all text and user_ids for one label
            texts_ids = df[df[label_col] == label][[text_col, user_id]]
            indices = texts_ids.index
            for i in indices:
                words = tknzr.tokenize(texts_ids.loc[i, text_col])
                # build word count dict
                for word in words:
                    if isalpha == True: 
                        # append only if word is all characters (no numbers, hashtags, urls, contractions like can't)
                        if word.isalpha():
                            if word in self._word_count_dict[label]:
                                self._word_count_dict[label][word] += 1 
                                self._word_count_dict['total'][word] += 1 #for calculation
                            else:
                                self._word_count_dict[label][word] = 1
                                if word in self._word_count_dict['total']:
                                    self._word_count_dict['total'][word] += 1
                                else:
                                    self._word_count_dict['total'][word] = 1
                    # append all words regardless of presence of non-characters
                    else: 
                        if word in self._word_count_dict[label]:
                            self._word_count_dict[label][word] += 1 
                            self._word_count_dict['total'][word] += 1 #for calculation
                        else:
                            self._word_count_dict[label][word] = 1
                            if word in self._word_count_dict['total']:
                                self._word_count_dict['total'][word] += 1
                            else:
                                self._word_count_dict['total'][word] = 1
        return None
    
    def _get_scores(self):
        """
        builds:
            score for each word for each label:
            self._score_dict = {
            'label_1': {
                'word_1': .32,
                'word_2': .87, ...
                }, ...
            }
        """
        for label_key in self._word_count_dict.keys():
            if label_key != 'total':
                self._score_dict[label_key] = {}
                for word_key in self._word_count_dict[label_key].keys():
                    # for each word: count of occurrence in class / count of occurrence all classes
                    self._score_dict[label_key][word_key] = self._word_count_dict[label_key][word_key] / self._word_count_dict['total'][word_key]
        return None
    
    def _get_top_k(self, k, thresh):
        """
        inputs:
            k: top k proto-words
            thresh: only words occurring >= thresh to be considered
                put 1 if no threshold desired
        builds:
            top k protowords for each label:
            self._k_dict = {
            'label_1': [
                ('protoword_1', 1.0),
                ('protoword_2', .93), ...
                ], ...
            }
        """
        for label_key in self._score_dict.keys():
            self._k_dict[label_key] = []
            # sort by value (score for the word) in descending order
            sorted_word_score_list = sorted(self._score_dict[label_key].items(), key=operator.itemgetter(1))
            sorted_word_score_list.reverse()

            # for each word
            for pair in sorted_word_score_list:
                # append only if word occurrence in class is above threshold
                if self._word_count_dict[label_key][pair[0]] >= thresh:
                    # append only top k
                    while len(self._k_dict[label_key]) < k:
                        self._k_dict[label_key].append(pair)
                        break
        return None


class protoHashtags():
    """
    Get the k most prototypical hashtags for each class (label)
    Functions the same way as protoWords class
    """
    def __init__(self):
        self._ht_count_dict = {'total': {}}
        self._score_dict = {}
        self._k_dict = {}
        
    def fit(self, df, label_col, ht_col, user_id):
        self._get_hashtag_count(df, label_col, ht_col, user_id)
        self._get_scores()
        
    def top_k(self, k, thresh):
        self._get_top_k(k, thresh)
        return self._k_dict 

    def _get_hashtag_count(self, df, label_col, ht_col, user_id):
        # labels (classes)
        labels = df[label_col].unique()
        for label in labels:
            self._ht_count_dict[label] = {}

            # all htags and user_ids for one label
            ht_ids = df[df[label_col] == label][[ht_col, user_id]]
            indices = ht_ids.index
            for i in indices:
                htags = ht_ids.loc[i, ht_col]

                # build ht count dict
                for ht in htags:
                    ht = ht.lower()
                    if ht in self._ht_count_dict[label]:
                        self._ht_count_dict[label][ht] += 1
                        self._ht_count_dict['total'][ht] += 1
                    else:
                        self._ht_count_dict[label][ht] = 1
                        self._ht_count_dict['total'][ht] = 1
        return None

    def _get_scores(self):
        for label_key in self._ht_count_dict.keys():
            if label_key != 'total':
                self._score_dict[label_key] ={}
                for ht_key in self._ht_count_dict[label_key].keys():
                    self._score_dict[label_key][ht_key] = self._ht_count_dict[label_key][ht_key] / self._ht_count_dict['total'][ht_key]
        return None

    def _get_top_k(self, k, thresh):
        for label_key in self._score_dict.keys():
            self._k_dict[label_key] = []
            sorted_ht_score_list = sorted(self._score_dict[label_key].items(), key=operator.itemgetter(1))
            sorted_ht_score_list.reverse()

            for pair in sorted_ht_score_list:
                if self._ht_count_dict[label_key][pair[0]] >= thresh:
                    while len(self._k_dict[label_key]) < k:
                        self._k_dict[label_key].append(pair)
                        break
        return None


def create_protoword_features(df, proto_word_obj, text_col, user_id, tok_type, isalpha, k, thresh):
    """
    inputs: 
        df: text must be aggregated by user
        proto_word_obj: must fitted protoWords object
        text is one long string
        see protoWords class for more details
    output:
        dataframe where each user is one row
        each protoword in each class is a feature and the score is the value        
        join with original df using user_id as key
    """
    orig_col = list(df.columns)
    orig_col.remove(user_id)
    
    # fit on parameters
    class_k_word_dict = proto_word_obj.top_k(k, thresh)
    
    if tok_type == 'full':
        tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False)
    else: # default is super clean
        tknzr = nltk.tokenize.TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    
    # create features per user
    indices = df.index
    for i in indices:
        words = tknzr.tokenize(df.loc[i, text_col])   
        for label_key in class_k_word_dict.keys():
            sum_num = 0 
            # word_key_pair is a tuple like ('emergency', 0.42)
            for word_key_pair in class_k_word_dict[label_key]:
                # count of specific proto-word used by the user
                num = words.count(word_key_pair[0])
                # total count of label's proto-words used by the user
                sum_num += num
                
                if isalpha == True:
                    denom = len([word for word in words if word.isalpha()])
                else:
                    denom = len(words)                
                # score for proto-word feature
                try:
                    df.loc[i, word_key_pair[0]] = num/denom
                except:
                    df.loc[i, word_key_pair[0]] = 0
            # score for general label proto-words feature
            try:        
                df.loc[i, 'PROTO_WORD_SCORE_' + label_key] = sum_num/denom
            except:
                df.loc[i, 'PROTO_WORD_SCORE_' + label_key] = 0

    df = df.drop(orig_col, axis=1)
    return df


def create_protohashtag_features(df, proto_hash_obj, ht_col, user_id, k, thresh):
    """
    inputs: 
        df: hashtags must be aggregated by user
        proto_hash_obj: must be protoHashtags object
        hashtags are in a list
        see protoWords class for more details
    output:
        dataframe where each user is one row
        each protohashtag in each class is a feature and the score is the value        
        join with original df using user_id as key
    """
    orig_col = list(df.columns)
    orig_col.remove(user_id)

    # fit on parameters
    class_k_ht_dict = proto_hash_obj.top_k(k, thresh)
    
    # create features per user
    indices = df.index
    for i in indices:
        hashtag_lst = df.loc[i, ht_col]
        hashtag_lst = [ht.lower() for ht in hashtag_lst] # work with lowercase only
        for label_key in class_k_ht_dict.keys():
            sum_num = 0 
            # word_key_pair is a tuple like ('emergency', 0.42)
            for word_key_pair in class_k_ht_dict[label_key]:
                # count of specific proto-hashtag used by the user
                num = hashtag_lst.count(word_key_pair[0])
                # total count of label's proto-hashtags used by the user
                sum_num += num
                # count of total hashtags by user
                denom = len(hashtag_lst)
                
                # score for proto-hashtag feature
                try:
                    df.loc[i, word_key_pair[0]] = num/denom       
                except:
                    df.loc[i, word_key_pair[0]] = 0
            # score for general label proto-hashtag feature
            try:
                df.loc[i, 'PROTO_HT_SCORE_' + label_key] = sum_num/denom
            except:
                df.loc[i, 'PROTO_HT_SCORE_' + label_key] = 0
                    
    df = df.drop(orig_col, axis=1)
    return df

