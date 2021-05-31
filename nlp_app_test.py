import pandas as pd
import nltk
#nltk.download('punkt')
from nltk import sent_tokenize
import transformers
import streamlit as st
from pipelines import pipeline as p
from streamlit.components.v1 import html

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

# sidebar panel
def s_p(text, unsafe_allow_html=True):
    tmp = st.sidebar.markdown(text, unsafe_allow_html=True)
    return tmp
s_p("<h2>Who I am?</h2>")
s_p("<i>A learner who learns from everywhere to share what meants to be shared</i>")
s_p("<br></br>")
s_p("<h3> Pay me with your feedback</h3>")

#main panel
st.title("Generate questions fom Texts")
st.markdown("**Enter any paragraph of the books/webpage/snippets to get framed questions on fly**")
quest = st.text_area("Enter Text Here", height=250, max_chars = 3500)
st.markdown("        ")

if st.button("Run and Wait"):
    nlp = load_module()
    st.markdown("Model loaded")

    def output():
        ls_out = []
        for i in split_doc(quest):
            q = nlp(i)
            print(q)
            if (len(q) == 1) or (len(q) <1):
                q = [q]
            ls_out.append(q)
        return sum(ls_out,[])
    ls_out = output()
    st.write(ls_out)
st.markdown("")
st.markdown("**Click below to connect me**", unsafe_allow_html=True)
html("""<a href="https://twitter.com/intent/tweet?screen_name=neural_digger&ref_src=twsrc%5Etfw" class="twitter-mention-button" data-show-count="false">Tweet to @neural_digger</a><script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>""")
    