# import streamlit as st
# import pickle
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS

# # Sidebar contents
# with st.sidebar:
#     st.title("🤗💬 LLM Chat App")
#     st.markdown(
#         """
#     ## About
#     This app is an LLM-powered chatbot built using:
#     - [Streamlit](https://streamlit.io/)
#     - [LangChain](https://python.langchain.com/)
#     - [OpenAI](https://platform.openai.com/docs/models) LLM model

#     """
#     )
#     st.write("Made with ❤️ by Ayush Kumar")


# def main():
#     st.header("Chat with PDF 💬")
#     load_dotenv()
#     pdf = st.file_uploader("Upload your PDF", type="pdf")

#     st.write(pdf.name)
#     if pdf is not None:
#         pdf_reader = PdfReader(pdf)

#         text = ""
#         for page in pdf_reader.pages:
#             text += page.extract_text()

#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000, chunk_overlap=200, length_function=len
#         )
#         chunks = text_splitter.split_text(text=text)

#         # Embedding
#         embeddings = OpenAIEmbeddings()
#         VectorStore = FAISS.from_texts(text=text)
#         store_name = pdf.name[:-4]
#         with open(f"{store_name}.pkl", "wb") as f:
#             pickle.dump(VectorStore, f)
#         # st.write(chunks)
#         # st.write(text)


# if __name__ == "__main__":
#     main()

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)

            st.write(response)


if __name__ == "__main__":
    main()
