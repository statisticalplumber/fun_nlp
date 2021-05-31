import pandas as pd
import nltk
#nltk.download('punkt')
from nltk import sent_tokenize
import transformers
import streamlit as st
from pipelines import pipeline as p

@st.cache(allow_output_mutation=True)
def load_module():
    nlp = p("e2e-qg")
    return nlp

def split_doc(document):
    paragraphs = []
    for paragraph in document.replace("\r\n", "\n").split("\n\n"):
        if len(paragraph.strip()) > 0:
            paragraphs.append(sent_tokenize(paragraph.strip()))

    window_size = 3
    passages = []
    for paragraph in paragraphs:
        for start_idx in range(0, len(paragraph), window_size):
            end_idx = min(start_idx+window_size, len(paragraph))
            passages.append(" ".join(paragraph[start_idx:end_idx]))

#     print("Paragraphs: ", len(paragraphs))
#     print("Sentences: ", sum([len(p) for p in paragraphs]))
#     print("Passages: ", len(passages))
    return passages

quest = st.text_area("Enter Text Here", height=250)
st.markdown("        ")

if st.button("Click"):
    nlp = load_module()
    st.markdown("Model loaded")
    ls_out = []
    for i in split_doc(quest):
        q = nlp(i)
        print(q)
        if (len(q) == 1) or (len(q) <1):
            q = [q]
        ls_out.append(q)
    st.write(sum(ls_out,[]))