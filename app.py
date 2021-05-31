# loading packages
import pandas as pd
import streamlit as st
from transformers import T5ForConditionalGeneration,T5Tokenizer
# import spacy
# from spacy import displacy
from nltk import sent_tokenize
import pandas as pd
import numpy as np

########### imp code snippet ##################

def get_question(sentence,answer):
  text = "context: {} answer: {} </s>".format(sentence,answer)
  #print (text)
  max_len = 256
  encoding = question_tokenizer.encode_plus(text,max_length=max_len, pad_to_max_length=True, return_tensors="pt")

  input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

  outs = question_model.generate(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  early_stopping=True,
                                  num_beams=5,
                                  num_return_sequences=1,
                                  no_repeat_ngram_size=2,
                                  max_length=200)


  dec = [question_tokenizer.decode(ids) for ids in outs]


  Question = dec[0].replace("question:","")
  Question= Question.strip()
  return Question.replace('<pad>  ', '').replace("</s>", "")

# spacy section
  

def check_verb(token):
    """Check verb type given spacy token"""
    if token.pos_ == 'VERB':
        indirect_object = False
        direct_object = False
        for item in token.children:
            if(item.dep_ == "iobj" or item.dep_ == "pobj"):
                indirect_object = True
            if (item.dep_ == "dobj" or item.dep_ == "dative"):
                direct_object = True
    else:
        return [token.pos_, token]

		
def get_noun(text):
    doc = nlp(text)
    def check_verb(token):
        """Check verb type given spacy token"""
        if token.pos_ == 'VERB':
            indirect_object = False
            direct_object = False
            for item in token.children:
                if(item.dep_ == "iobj" or item.dep_ == "pobj"):
                    indirect_object = True
                if (item.dep_ == "dobj" or item.dep_ == "dative"):
                    direct_object = True
        else:
            return [token.pos_, token]
    out = [check_verb(t) for t in doc]
    ls_noun = []
    for i in out:
        if i != None:
            if i[0] == 'NOUN':
                ls_noun.append(i[1])
    return ls_noun
	
	
def split_doc(document):
    paragraphs = []
    for paragraph in document.replace("\r\n", "\n").split("\n\n"):
        if len(paragraph.strip()) > 0:
            paragraphs.append(sent_tokenize(paragraph.strip()))

    window_size = 2
    passages = []
    for paragraph in paragraphs:
        for start_idx in range(0, len(paragraph), window_size):
            end_idx = min(start_idx+window_size, len(paragraph))
            passages.append(" ".join(paragraph[start_idx:end_idx]))

#     print("Paragraphs: ", len(paragraphs))
#     print("Sentences: ", sum([len(p) for p in paragraphs]))
#     print("Passages: ", len(passages))
    return passages

def question_derived(document):
    #print(document)
    ls_quest = []
    ls_ans = []
    for doc in split_doc(document):
        ls_quest.append(get_question(doc, get_noun(doc)[:1][0]))
        ls_ans.append(get_noun(doc)[:1][0].text)
    return ls_quest, ls_ans



st.sidebar.markdown('<h3> Generate Questions </h3>', unsafe_allow_html = True)

st.title("Basic Question Generator App")

@st.cache(allow_output_mutation=True)
def load_module():
    # nlp = spacy.load("en_core_web_sm")
    question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
    question_tokenizer = T5Tokenizer.from_pretrained('t5-base')

    return question_model, question_tokenizer



quest = st.text_area("Enter Text Here")
st.markdown("        ")
st.markdown("Enter keywords with : seprator around questions need to be generated")
exp_ans = st.text_input("Example:  keyword1:keyword2:keyword3")

if st.checkbox("True"):
    question_model, question_tokenizer = load_module()
    if st.button("Run"):
        passages = pd.Series(split_doc(quest))

        ls_quest = []
        ls_ans = []
        for i in exp_ans.split(":"):
            content = passages[passages.str.contains(i)].iloc[0]
            ls_quest.append(get_question(content, i))
            ls_ans.append(i)

        ans = pd.DataFrame({'Question': ls_quest, 'Answer': ls_ans})

        st.write(ans.to_dict(orient='records'))