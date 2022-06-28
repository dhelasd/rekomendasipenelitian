import warnings
import itertools
from collections import Counter
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import string

import streamlit as st

from prefixspan import PrefixSpan

import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
nltk.download('stopwords')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
listStopword =  set(stopwords.words('indonesian')) 

st.title(f'Roadmaps Penelitian Berdasarkan Pattern Mining Jurnal Ilmiah')

file = st.sidebar.file_uploader("Choose a file")

ori = st.sidebar.checkbox('Data Original')
nlp = st.sidebar.checkbox('Data NLP')

def read_data(x,file):
    XL = pd.ExcelFile(file)
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

if file is not None:
    sheet_opt = pd.ExcelFile(file).sheet_names
    sheet_dict = {n:m for m,n in enumerate(sheet_opt)}
    x = st.sidebar.selectbox(
        "Pilih Jurusan",
        tuple([o for o in sheet_opt if o!='LPPM'])
    )

    data, sheet= read_data(sheet_dict[x], file)


    thn = st.sidebar.multiselect(
        "Pilih Tahun",
        tuple([o for o in sorted(data['Tahun'].unique())]),
    )
    if thn and len(thn)>1:
        data = data[data['Tahun'].isin(thn)]
    elif thn and len(thn)==1:
        data = data[data['Tahun']==thn[0]]

    txt = st.sidebar.text_area('Hapus Teks', 
    '''bengkulu, and, of, on, in, based, the, to, indonesia, thailand, from, for, berbasis, muara bangkahulu, kota, di, untuk, sungai serut, sawah lebar, kandang limun, lempuing ''')

    ms = st.sidebar.slider("PrefixSpan - Minimum Frequencies",min_value=2,max_value=25, value=3, step=1)
    mp = st.sidebar.slider("PrefixSpan - Maximum Pattern Length",min_value=1,max_value=25, value=10, step=1)
    mnp = st.sidebar.slider("PrefixSpan - Minimum Pattern Length",min_value=1,max_value=10, value=3, step=1)
    listtoberemoved = txt.split(', ')

    topik = list(set(itertools.chain.from_iterable(data['Topik'].str.split(', ').values.tolist())))
    df, stemmed = preprocessing(data, listtoberemoved)

    pf = pd.DataFrame()
    while pf.empty and mnp!=0:
        msx=ms
        while ms>0:
            try:
                pf = mining(data, stemmed, ms=ms, mp=mp, mnp=mnp)
                break
            except:
                ms = ms-1
        mnp = mnp-1

    st.header(f"Jurusan {x}")

    rekomendasi = dict(Counter([n for n in ', '.join(pf['sequence'].values.tolist()).split(', ') if n in topik])).keys()

    st.subheader(f"Rekomendasi Bidang Penelitian : {', '.join(rekomendasi)}")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Kata yang Sering Muncul")
        wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color='white')
        wordcloud.generate(','.join(df['stemmed'].values.tolist()))
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig)
    with col2:
        st.subheader("Komposisi Topik Jurnal Ilmiah")
        fig = px.pie(data.groupby("Topik").size().reset_index().rename(columns={0:'Jumlah'}), names='Topik',values='Jumlah')
        st.plotly_chart(fig, use_container_width=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        bidang = st.selectbox(
            "Bidang Penelitian",
            tuple(rekomendasi)
        )
    if msx != ms:
        st.write(f"Data Insufficient, Minimum Frequency = {ms}")
    try:
        pfr = pf[[bidang in n for n in pf['sequence']]]

        if len(pfr)>=10:
            def_head = 10
        else:
            def_head = len(pfr)

        if len(pfr)>1:
            with col6:
                head = st.slider("Pilih Jumlah Data",min_value=1,max_value=len(pfr), value=def_head, step=1)
        else:
            head = 10

        col3, col4 = st.columns(2)
        with col3:
            st.subheader(f"Top {head} Sequential Pattern")
            fig = px.bar(pfr.head(head),x='sequence', y='freq')
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            st.subheader("Keywords Perencanaan Roadmap Penelitian")
            wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color='lightgray')
            wordcloud.generate(','.join(pfr['sequence'].head(head).values.tolist()))
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(fig)
        
        if ori:
            st.subheader("Data Original")
            st.write(data)
        
        if nlp:
            st.subheader("Natural Language Processing")
            st.write(df)
    except:
        st.write("Data insufficient, bidang riset tidak ditemukan dalam sequential Pattern")
