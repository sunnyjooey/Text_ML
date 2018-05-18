import pandas as pd
import nltk
import stanford


def parse_svo(df, col):
    """
    Parses text to Subject-Verb-Object triples
    Input `col` needs to be list of tokenized texts (list of lists)
    """
    # for index, row in df.iterrows():
        # df.loc[index, 'parse_sents'] = list(stanford.parser.parse_sents(row[col]))
        # df.loc[index, 'parse_sentences'] = [list(i) for i in df.loc[index,'parse_sents']]
        # df.loc[index, 'svos'] = [get_SVOs_in_sentence_tree(i[0]) for i in df.loc[index,'parse_sentences']]
        # print(index)
    df['parse_sents'] = df[col].apply(lambda x: list(stanford.parser.parse_sents(x)))
    df['parse_sents'] = df['parse_sents'].apply(lambda x: [list(i) for i in x])
    df['svos'] = df['parse_sents'].apply(lambda x: [get_SVOs_in_sentence_tree(i[0]) for i in x])
    return df

def extract_noun(tree):
    """
    Intended to be used as being passed a NP tag as part of a subject or direct object
    component of a sentence. Looks for the most directly related NN (noun) or PRP (pronoun)
    searching recursively, depth-first.
    """
    if not tree.label().startswith('N'):
        return ''
    result = ''
    for child in tree:
        if child.label() == 'NP':
            result += extract_noun(child)
        elif child.label().startswith('NN') or child.label() == 'PRP':
            result += child[0] + ' '
        elif child.label() == ',':
            break
    return result

def extract_verb(tree, lem_verb=True):
    """
    Intended to receive a VP predicate, looks for the Verb. If a compound verb, returns the
    concatenation of all the verbs. Able to lemmatize verbs using the passed lemmatizer.
    """
    if not tree.label().startswith('V'):
        return ''
    lemmer = nltk.stem.WordNetLemmatizer()
    result = ''
    for child in tree:
        if child.label() == 'VP':
            result += extract_verb(child, lem_verb=lem_verb)
        elif child.label().startswith('VB'):
            if lem_verb:
                result += lemmer.lemmatize(child[0], pos='v') + ' '
            else:
                result += child[0] + ' '
    return result

def extract_direct_object(tree):
    """
    Extracts the direct object from a simple declarative sentence VP predicate.
    Attempts to even bypass a prepositional predicate to find the noun directly related to 
    the verb.
    """
    #pdb.set_trace()
    if not tree.label().startswith('VP'):
        return ''
    for child in tree:
        if child.label() == 'NP':
            return extract_noun(child)
        elif child.label() == 'PP':
            for subchild in child:
                if subchild.label() == 'NP':
                    return extract_noun(subchild)
    for child in tree:
        if child.label() == 'VP':
            return extract_direct_object(child)
    return ''

def extract_SVO(tree, last_noun=True, lem_verb=True):
    """
    Extracts an SVO triple from a simple declarative sentence (S) tree.
    last_noun: True if only the last word from the noun and verb is required
    """
    subject = verb = objec = ''
    if not tree.label() == 'S':
        return None
    for child in tree:
        if child.label() == 'NP':
            subject = extract_noun(child)
        elif child.label() == 'VP':
            verb = condense_verbs(extract_verb(child, lem_verb))
            objec = extract_direct_object(child)
    if last_noun:
        subject = subject.strip().split(' ')[-1]
        objec = objec.strip().split(' ')[-1]
    return (subject, verb, objec)


def get_SVOs_in_sentence_tree(tree):
    """
    Locates simple declarative sentences in a tree and returns a list of all the SVO triples
    """
    result = []
    for subt in tree.subtrees():
        if subt.label()=='S':
            result.append(extract_SVO(subt, last_noun=True, lem_verb=True))
    return result

def condense_verbs(verbs):
    """
    for complex verbs, eliminate the auxiliar verbs
    """
    vblist = verbs.split()
    if len(vblist) < 2:
        return verbs
    aux= ['have', 'has', 'had', 'been', 'are', 'is', 'be', 'am', 'does', 'did', 'was', 'being', 'having']
    return " ".join([vb for vb in vblist if vb not in aux])