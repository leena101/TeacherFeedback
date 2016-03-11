
'''
B.Tech Major Project:
Automated Student Feedback based on Natural Language Processing and 
Opinion Mining
'''

'''
Imports
'''
import pandas,nltk    

'''
Process feedback data from a csv file.
Presently using dummy data.
'''
def readcsv():
    colnames = ['Teacher','Subject','Feedback']

    return pandas.read_csv('../data/feedback.csv', names=colnames)

'''
After reading csv file store data in a dataframe, by sorting it according to
each teacher and subject's unique combination.
'''
def makedataframe(data):
    df = data.groupby(['Teacher','Subject'])['Feedback'].unique().reset_index()

    #Introduced new column Tagged Feedback to store the unique feedback
    df.insert(3,'TaggedFeedback',0)

    #Allow TaggedFeedback to be overwritten with new values, other than 0
    pandas.options.mode.chained_assignment = None

    return df

'''
POS Tagging of unique feedback made by makedataframe after removing stopwords
which is stored in TaggedFeedback column which was made in makedataframe()
function.
'''
def postagger(df):
    count = 0

    txt = open('../data/stopwords.txt')
    stop = txt.read()

    for f in df['Feedback']:
        arr = []
        for a in f:
            b = nltk.pos_tag(a.split())
            for c in b:
                if c[0] not in stop:
                    arr.append(c)
        df['TaggedFeedback'][count] = arr
        count = count + 1

    return df
    

data = readcsv()
df = makedataframe(data)
df = postagger(df)
print (df.TaggedFeedback[1])
