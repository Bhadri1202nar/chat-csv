from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


def main():
    # Retrieve the API key from environment variables
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Check if the API key is loaded
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found in environment. Please set it in your .env file.")
        return

    # Initialize Google Gemini model
    try:
        gemini_model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"Error initializing Google's model:{e} \n Make sure you have GOOGLE_API_KEY set correctly" )
        return


    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:

        agent = create_csv_agent(
            gemini_model, csv_file, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,allow_dangerous_code=True)


        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question is not None and user_question != "":
            with st.spinner(text="In progress..."):
                try:
                    st.write(agent.run(user_question))
                except Exception as e:
                    st.error(f"Error running agent:{e}")


if __name__ == "__main__":
    main()