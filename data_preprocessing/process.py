import pandas as pd
import json
import nltk
from ttp import ttp # https://github.com/edburnett/twitter-text-python

def process_text(file_lst, label_lst, truncate=True):
    """
    Input: 
        file_lst: list of filenames to process
        label_lst: list of labels (in same order as file_lst)
        truncate: keep only text-related columns
    Output:
        dataframe with retweet/reply handles, hashtags, urls parsed out to separate columns        
    """
    raw_df = concat_tweet_sets(file_lst, label_lst, truncate=truncate)
    pars_df = parse_clean_text(raw_df)
    pars_df['tokenized_text'] = pars_df['rough_clean_text'].apply(lambda x: nltk.word_tokenize(x.lower().replace('(','').replace(')',''))) 
    return pars_df


def read_twitter_data(filename, label, truncate=True):
    """
    label is the label you want to give to the twitter data 
    output file is dataframe where each tweet is one row
    """
    # read in tweets
    tweets = []
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            t = json.loads(line)
            tweets.append(t)
            
    # make into datafram
    tweet_df = pd.DataFrame(tweets)
    
    #add label
    tweet_df['label'] = list([label]) * tweet_df.shape[0]
    
    #get only the text and user-object of the tweet (no user stats will be included)
    if truncate == True:
        tweet_df = tweet_df[['full_text', 'user', 'label']]
    return tweet_df
    

def concat_tweet_sets(filename_lst, label_lst, truncate=True):
    """
    filename_lst and label_lst must be in the same order
    """
    df = pd.DataFrame()
    # concatenate the files
    for i, file in enumerate(filename_lst):
        one_set = read_twitter_data(file, label_lst[i], truncate=truncate)
        df = pd.concat([df, one_set])
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    return df


def parse(text):
    """
    uses ttp package to parse out user handles, hashtags, urls, html from tweet's text
    """
    p = ttp.Parser()
    result = p.parse(text)
    users = result.users
    tags = result.tags
    urls = result.urls
    html = result.html
    return [users, tags, urls, html]


def clean_text(row):
    """
    cleans the tweet's text of user handles, urls, and hashtags (just the #)
    use for topic modeling
    """
    text = row['full_text']
    for user in row['to_users']:
        retweet_mark = 'RT @' + user + ':'
        reply_mark = '@' + user
        if retweet_mark in text:
            text = text.replace(retweet_mark, '')
        if reply_mark in text:
            text = text.replace(reply_mark, '')

    for url in row['urls']:
        if url in text:
            text = text.replace(url, '')
    #Get rid of hashtags        
    text = text.replace('#', '')
    return text
        

def parse_clean_text(df, truncate=True):
    """
    clean and parse the dataframe
    """
    # parse the tweet 
    df['parse_list'] = df['full_text'].apply(parse) # comes in list of lists
    parse_df = pd.DataFrame(df.parse_list.values.tolist()) # separate out into columns   
    parse_df.columns = ['to_users', 'hashtags', 'urls', 'html']
    
    # join together
    df = df.join(parse_df)
    df.drop(['parse_list'], axis=1, inplace=True)
    
    # get user_id
    df['user_id'] = df['user'].apply(lambda x: x['id'])
    
    # clean the text of retweet, reply, url, '#' into a new column
    df['clean_text'] = df.apply(clean_text, axis=1)
    
    # clean the text of only the '@', 'RT @', and '#' into a new column
    df['rough_clean_text'] = df['full_text'].apply(lambda x: x.replace('#','').replace('RT @','').replace('@', ''))
    
    # drop the user-objects from df
    if truncate == True:
        df.drop(['user'], axis=1, inplace=True)
    return df


def aggregate_by_user(df, user_id, text_col_list, tok_text_list, parse_col_list, truncate=True):
    """
    aggregate the tweets by user id
    text_col_list: list of columns of texts to aggregate
    tok_text_list: list of columns of text lists (e.g. tokenized text) to aggregate into a list
    parse_col_list: list of columns of parsed entities (hashtags, etc, in a list) to aggregate
    """
    for text_col in text_col_list:
        # join all texts with one space
        df[text_col + "_agg"] = df.groupby([user_id])[text_col].transform(lambda x: ' '.join(x))
    
    for tok_text_col in tok_text_list:
        agg_df = pd.DataFrame(df.groupby([user_id])[tok_text_col].apply(list)).reset_index()
        agg_df.rename(columns={tok_text_col: tok_text_col + '_agg'}, inplace=True)
        df = pd.merge(df, agg_df, how='left', on=['user_id'])
        
    for parse_col in parse_col_list:
        # aggregate lists into one list
        parse_df = df.groupby([user_id]).agg({parse_col: 'sum'}).reset_index()
        parse_df.rename(columns={parse_col: parse_col + '_agg'}, inplace=True)
        # left merge
        df = pd.merge(df, parse_df, how='left', on=['user_id'])
    
    # drop duplicates
    df = df.drop_duplicates(subset='user_id')
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)
        
    if truncate == True:
        # drop columns in text_col_list and parse_col_list 
        df.drop(text_col_list+parse_col_list+['rough_clean_text','tokenized_text'], axis=1, inplace=True)  
    return df