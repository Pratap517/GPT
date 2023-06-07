import streamlit as st
from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

os.environ["OPENAI_API_KEY"] = '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'

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.save_to_disk('index.json')

    return index

directory_path = "docs"
if not os.path.exists(directory_path):
    st.error("Directory 'docs' not found. Please make sure the directory exists.")
else:
    index = construct_index(directory_path)

def chatbot(input_text):
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    response = index.query(input_text, response_mode="compact")
    return response.response

# Add CSS style

if "messages" not in st.session_state:
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
        
        
page_bg_img = f"""
<style>


[data-testid="main.css-uf99v8.e1g8pov65"] {{
background-color: #F0F8FF;
}}

[data-testid="stSidebar"] {{
background-color: #F0F8FF;
border: 3px solid #CCDCCC;
border-radius: 2px
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


st.header("## Mjethwani Custom-trained AI Chatbot")
st.image("./yyy.jpeg", width=100)

with st.sidebar:
        
        
    user_input = st.text_input("#Your message: ", key="user_input")

    # handle user input
    if user_input:
        with st.spinner("Thinking..."):
            response = chatbot(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))
        st.session_state.messages.append(AIMessage(content=response.content))

# display message history
messages = st.session_state.get('messages', [])
for i, msg in enumerate(messages[1:]):
    if i % 2 == 0:
        message(msg.content, is_user=True, key=str(i) + '_user')
    else:
        message(msg.content, is_user=False, key=str(i) + '_ai')


# if st.button("Submit"):
#     response = chatbot(input_text)
#     st.text_area("Output", value=response)


