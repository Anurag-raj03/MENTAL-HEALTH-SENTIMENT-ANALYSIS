import streamlit as st
import numpy as np
import joblib as j
from bs4 import BeautifulSoup
import re
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import speech_recognition as sr
import warnings 
warnings.filterwarnings('ignore')

model = j.load(open('model_jbl', 'rb'))
cv=j.load(open('cv_jbl', 'rb'))
def removal(q):
    
    q = str(q).lower().strip()

    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    q = q.replace('[math]', '')
    
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")
    
    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()
    
    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    
    return q
def Duplicate_checker(q1,q2):
    t=[]
    q1=removal(q1)
    q2=removal(q2)

    #basic_feature
    t.append(len(q1))
    t.append(len(q2))

    t.append(len(q1.split(' ')))
    t.append(len(q2.split(' ')))
    
    def cm (x1,x2):
        x1=set(x1.lower().split(' '))
        x2=set(x2.lower().split(' '))
        cmn=x1.intersection(x2)
        return cmn
    
    t.append(len(cm(q1,q2)))

    def ttl(x1,x2):
        x1=set(x1.split(' '))
        x2=set(x2.split(' '))
        t_wrd=(len(x1)+len(x2))
        return t_wrd
    
    t.append(len(cm(q1, q2)))
    t.append(round(len(cm(removal(q1), removal(q2))) / ttl(removal(q1), removal(q2))) if ttl(removal(q1), removal(q2)) != 0 else 0)





    #fetching Token Feature
    def cwc_min(x1,x2):
        x1=len(x1.split(' '))
        x2=len(x2.split(' '))
        t=min(x1,x2)
        return(t)
    
    common_word_count = len(cm(q1, q2))
    minimum_word_count = cwc_min(q1, q2)
    
    t.append(common_word_count / minimum_word_count if minimum_word_count != 0 else 0)

    def cwc_max(x1,x2):
        x1=len(x1.split(' '))
        x2=len(x2.split(' '))
        t=max(x1,x2)
        return(t)
    
    max_word_count = cwc_max(q1, q2)
    t.append(common_word_count / max_word_count if max_word_count != 0 else 0)

    def st(x):
        st=set(stopwords.words('english'))
        wrd=x.lower().split(' ')
        sts=[i for i in wrd if i in st]
        return len(sts)
    
    def stp(x):
        st=set(stopwords.words('english'))
        wrd=x.lower().split(' ')
        sts=[i for i in wrd if i in st]
        return sts
    
    def cmn_stp(x1, x2):
        x1_stops = stp(x1)
        x2_stops = stp(x2)
        common_stops = {word for word in x1_stops if word in x2_stops}
        return len(common_stops)


    
    def csc_min(x1,x2):
        x1=len(stp(x1))
        x2=len(stp(x2))
        r=min(x1,x2)
        return r
    
    def csc_max(x1,x2):
        x1=len(stp(x1))
        x2=len(stp(x2))
        r=max(x1,x2)
        return r
    


    t.append(cmn_stp(q1,q2)/csc_min(q1,q2) if csc_min(q1,q2) !=0 else 0)
    t.append(cmn_stp(q1,q2)/csc_max(q1,q2) if csc_max(q1,q2) !=0 else 0)  


    def tkn_c(x):
     tkn = set(x.lower().split(' '))
     return tkn

    def cmn_tk(x1, x2):
     x1_tkn = tkn_c(x1)    
     x2_tkn = tkn_c(x2) 
     ct = x1_tkn.intersection(x2_tkn)
     return ct

    def cmn_t(x1, x2):
        x1_len = len(tkn_c(x1))    
        x2_len = len(tkn_c(x2)) 
        to = x1_len + x2_len
        return to

    def ctc_min(x1, x2):
        x1_len = len(tkn_c(x1))    
        x2_len = len(tkn_c(x2))
        te = min(x1_len, x2_len) 
        return te

    def ctc_max(x1, x2):
        x1_len = len(tkn_c(x1))    
        x2_len = len(tkn_c(x2))
        tn = max(x1_len, x2_len) 
        return tn

    t.append(cmn_t(q1, q2) / ctc_min(q1, q2) if ctc_min(q1, q2) != 0 else 0)
    t.append(cmn_t(q1, q2) / ctc_max(q1, q2) if ctc_max(q1, q2) != 0 else 0)

    
    def stc(x1,x2):
        x1=x1.lower().split(' ')
        x2=x2.lower().split(' ')
        if x1[0]==x2[0]:
            kt=1
        else:
            kt=0
        return kt 

    def lste(x1,x2):
        x1=x1.lower().split(' ')
        x2=x2.lower().split(' ')
        if x1[-1]==x2[-1]:
            mt=1
        else:
            mt=0
        return mt    

    t.append(stc(q1,q2))  
    t.append(lste(q1,q2)) 



    #length Base feature
    def mean(x1,x2):
        x1=len(x1.split(' '))
        x2=len(x2.split(' '))
        mt=(x1+x2)/2
        return mt
    
    def w(x):
        srt=len(x.split(' '))
        return srt
    
    def diff(x1,x2):
        x1=w(x1)
        x2=w(x2)
        t=abs(x1-x2)
        return t

    
    def lstr(x1, x2):

      def lcstm(s1, s2):
          m = len(s1)
          n = len(s2)
          dp = [[0] * (n + 1) for _ in range(m + 1)]
          max_len = 0

          for i in range(1, m + 1):
              for j in range(1, n + 1):
                  if s1[i - 1] == s2[j - 1]:
                      dp[i][j] = dp[i - 1][j - 1] + 1
                      max_len = max(max_len, dp[i][j])
                  else:
                      dp[i][j] = 0

          return max_len


      common_substring_length =   lcstm(x1, x2)
      total_length = len(x1) + len(x2)


      ratio = 2 * common_substring_length / total_length if   total_length != 0 else 0

      return ratio
     

    t.append(mean(q1,q2))
    t.append(diff(q1,q2)) 
    t.append(lstr(q1,q2)) 


    #Fuzzy feature:
    def fr(x1,x2):
        x1=x1.split(' ')
        x2=x2.split(' ')
        sr=fuzz.ratio(x1,x2)
        return sr 
    
    def fpr(x1,x2):
        x1=x1.split(' ')
        x2=x2.split(' ')
        lsr=fuzz.partial_ratio(x1,x2)
        return lsr
    
    def fst(x1,x2):
        x1=x1.split(' ')
        x2=x2.split(' ')
        fsr=fuzz.token_sort_ratio(x1,x2)
        return fsr
    
    def tst(x1,x2):
        x1=x1.split(' ')
        x2=x2.split(' ')
        tsr=fuzz.token_set_ratio(x1,x2)
        return tsr
    
    t.append(fr(q1,q2))
    t.append(fpr(q1,q2))
    t.append(fst(q1,q2))
    t.append(tst(q1,q2))

    #Bg of word features

    q1_bgw = cv.transform([q1]).toarray()
    q2_bgw = cv.transform([q2]).toarray() 

    return np.hstack((np.array(t).reshape(1, 22), q1_bgw, q2_bgw))    
    





def con():
    r = sr.Recognizer()
    
    with sr.Microphone() as source:
        st.text("Asking  Question 1....")
        audio = r.listen(source)
        st.text("Recognizing now...")
    
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.text("Sorry, could not understand audio.")
        return None
    except sr.RequestError as e:
        st.text("Could not request results from Google Web Speech API; {0}".format(e))
        return None

def conv():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        st.text("Asking Your Question 2....")
        audio = r.listen(source)
        st.text("Recognizing now...")

    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        st.text("Sorry, could not understand audio.")
        return None
    except sr.RequestError as e:
        st.text("Could not request results from Google Web Speech API; {0}".format(e))
        return None


st.header("Duplicate Question Predictor")

# Initialize session state variables
if 'q1_speech' not in st.session_state:
    st.session_state.q1_speech = ""
if 'q2_speech' not in st.session_state:
    st.session_state.q2_speech = ""

ask1_button = st.button("Ask Question 1")


if ask1_button:
    q1_speech = con()
    st.session_state.q1_speech = q1_speech+str(" ? ")
    print("Question 1:", q1_speech)


q1_text = st.text_input("Enter Question 1", value=st.session_state.q1_speech)



ask2_button = st.button("Ask Question 2")
if ask2_button:
    q2_speech = conv()
    st.session_state.q2_speech = q2_speech+str(" ? ")
    print("Question 2:", q2_speech)


q2_text = st.text_input("Enter Question 2", value=st.session_state.q2_speech)

q1 = removal(q1_text)
q2 = removal(q2_text)

if st.button('Find'):
    question = Duplicate_checker(q1, q2)
    question_fe = np.array(question).reshape(1, -1)
    result = model.predict(question_fe)[0]

    if result:
        st.header('Duplicate')
    else:
        st.header('Not Duplicate')


































# q1 = st.text_input('Enter Question 1')
# q2 = st.text_input('Enter Question 2')

# if st.button('Find'):
#     q1 = removal(q1)
#     q2 = removal(q2)
    
#     question = Duplicate_checker(q1, q2)
#     question_fe = np.array(question).reshape(1, -1)
#     result = model.predict(question_fe)[0]

#     if result:
#         st.header('Duplicate')
#     else:
#         st.header('Not Duplicate')
