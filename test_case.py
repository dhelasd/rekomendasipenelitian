import unittest
from matplotlib.pyplot import cla
from prefixspan import PrefixSpan
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import string
import os

factory = StemmerFactory()
stemmer = factory.create_stemmer()
listStopword =  set(stopwords.words('indonesian')) 

listtoberemoved = ['bengkulu', 'and', 'of', 'on','in','based','the','to','indonesia','thailand','from', 'for','berbasis']

def read_data(x):
    XL = pd.ExcelFile(os.path.join("data","data_mod.xlsx"))
    sheet = XL.sheet_names[x]
    data = pd.read_excel(XL, sheet)
    data = data[~data['Topik'].isnull()]
    data['Tahun'] = data['Tahun'].astype(int)
    data = data.reset_index(drop=True)
    data = data[data.columns.tolist()[:-3]].join(data[data.columns.tolist()[-2:]]).join(data[data.columns.tolist()[-3]])
    return data, sheet

def preprocessing(data, listtoberemoved):
    df = pd.DataFrame(data['Judul'],columns=['Judul']).copy()
    cleaned = []
    for n in df['Judul'].values:
        n = n.lower()
        n = re.sub(r':', '', n)
        n = re.sub(r'‚Ä¶', '', n)
        n = re.sub(r'[^\x00-\x7F]+',' ', n)
        n = re.sub('[^a-zA-Z]', ' ', n)
        n = re.sub("&lt;/?.*?&gt;","&lt;&gt;",n)
        n = re.sub("(\\d|\\W)+"," ",n)
        n = re.sub(r'â', '', n)
        n = re.sub(r'€', '', n)
        n = re.sub(r'¦', '', n)
        cleaned.append(n)
    df['cleaned'] = cleaned

    tokenized = []
    for n in cleaned:
        n = word_tokenize(n)
        tokenized.append(n)
    df['tokenized'] = [', '.join(n) for n in tokenized]

    removed = []
    for ts in tokenized:
        n = []
        for t in ts:
            if t not in listtoberemoved and t not in listStopword and t not in string.punctuation:
                n.append(t)
        removed.append(n)
    df['removed'] = [', '.join(n) for n in removed]

    stemmed = []
    for n in removed:
        n = ' '.join(n)
        n = stemmer.stem(n)
        n = n.split(' ')
        stemmed.append(n)
    df['stemmed'] = [' '.join(n) for n in stemmed]
    return df, stemmed

def mining(data,stemmed, ms, mp, mnp):

    dx = [n for n in [a + b + c for a,b,c in zip(stemmed,data['Keyword'].str.split(",").values.tolist(),data['Topik'].str.split(",").values.tolist())]]
    ps = PrefixSpan(dx)
    pf_results = pd.DataFrame(ps.frequent(ms), columns=['freq','sequence'])
    
    pf_results['sequence'] = [', '.join(n) for n in pf_results['sequence'].values.tolist()]
    pf_results = pf_results[[len(n)<=mp for n in pf_results['sequence'].str.split(",").values.tolist()]]
    pf_results = pf_results[[len(n)>=mnp for n in pf_results['sequence'].str.split(",").values.tolist()]].sort_values(by='freq',ascending=False).reset_index(drop=True)
    return pf_results

def run(seq,ms,mp,mnp):
    data, sheet= read_data(seq)
    df, stemmed = preprocessing(data, listtoberemoved)
    pf = mining(data, stemmed, ms=ms, mp=mp, mnp=mnp)
    return pf

class test_case(unittest.TestCase):
    
    def test_informatika(self):
        self.assertFalse(run(1,2,10,3).empty)
        self.assertTrue(read_data(1)[1]=="Informatika")
    
    def test_sipil(self):
        self.assertFalse(run(2,2,10,3).empty)
        self.assertTrue(read_data(2)[1]=="Sipil")
    
    def test_mesin(self):
        self.assertFalse(run(3,3,10,3).empty)
        self.assertTrue(read_data(3)[1]=="Mesin")
    
    def test_elektro(self):
        self.assertFalse(run(4,2,10,3).empty)
        self.assertTrue(read_data(4)[1]=="Elektro")
    
    def test_arsitektur(self):
        self.assertFalse(run(5,2,10,3).empty)
        self.assertTrue(read_data(5)[1]=="Arsitektur")
    
    def test_SI(self):
        self.assertFalse(run(6,2,10,3).empty)
        self.assertTrue(read_data(6)[1]=="Sistem Informasi")

if __name__ == '__main__':
    unittest.main()