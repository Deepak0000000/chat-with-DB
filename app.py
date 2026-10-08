import sqlite3
from pathlib import Path

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent


# ==================================================
# Streamlit Page Configuration
# ==================================================

st.set_page_config(
    page_title="LangChain: Chat with SQL Database",
    page_icon="🦜",
)

st.title("🦜 LangChain: Chat with SQL Database")


# ==================================================
# Database Constants
# ==================================================

LOCALDB = "LOCALDB"
MYSQL = "MYSQL"


# ==================================================
# Sidebar
# ==================================================

db_choice = st.sidebar.radio(
    "Choose Database",
    (
        "Use SQLite Database (student.db)",
        "Connect to MySQL Database",
    ),
)


# ==================================================
# MySQL Configuration
# ==================================================

if db_choice == "Connect to MySQL Database":

    db_uri = MYSQL

    mysql_host = st.sidebar.text_input(
        "MySQL Host",
        value="127.0.0.1",
    )

    mysql_port = st.sidebar.text_input(
        "Port",
        value="3306",
    )

    mysql_user = st.sidebar.text_input(
        "Username",
        value="root",
    )

    mysql_password = st.sidebar.text_input(
        "Password",
        type="password",
    )

    mysql_db = st.sidebar.text_input(
        "Database Name",
        value="student",
    )

else:

    db_uri = LOCALDB


# ==================================================
# Groq API Key
# ==================================================

api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password",
)

if not api_key:

    st.info("Please enter your Groq API Key.")

    st.stop()


# ==================================================
# LLM
# ==================================================

llm = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-20b",
    temperature=0,
    streaming=True,
)


# ==================================================
# Configure Database
# ==================================================

@st.cache_resource
def configure_db():

    # --------------------------------------------------
    # SQLite
    # --------------------------------------------------

    if db_uri == LOCALDB:

        db_path = Path(__file__).parent / "student.db"

        creator = lambda: sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
        )

        engine = create_engine(
            "sqlite://",
            creator=creator,
        )

        return SQLDatabase(engine)

    # --------------------------------------------------
    # MySQL
    # --------------------------------------------------

    else:

        connection_url = URL.create(
            drivername="mysql+mysqlconnector",
            username=mysql_user,
            password=mysql_password,
            host=mysql_host,
            port=int(mysql_port),
            database=mysql_db,
        )

        engine = create_engine(connection_url)

        # Test MySQL connection
        with engine.connect():
            pass

        return SQLDatabase(engine)


# ==================================================
# Database Connection
# ==================================================

try:

    db = configure_db()

    st.sidebar.success("✅ Database Connected")

except Exception as e:

    st.error(
        f"Database Connection Error\n\n{e}"
    )

    st.stop()


# ==================================================
# SQL Toolkit
# ==================================================

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=llm,
)

# ==================================================
# SQL Agent
# ==================================================

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type="tool-calling",
    verbose=True,
    max_iterations=5,
)


# ==================================================
# Chat History
# ==================================================

if (
    "messages" not in st.session_state
    or st.sidebar.button("Clear Chat")
):

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello 👋 Ask me anything "
                "about your SQL database."
            ),
        }
    ]


# ==================================================
# Display Chat History
# ==================================================

for message in st.session_state.messages:

    st.chat_message(
        message["role"]
    ).write(
        message["content"]
    )


# ==================================================
# Chat Input
# ==================================================

user_query = st.chat_input(
    "Ask a question..."
)


# ==================================================
# Process User Question
# ==================================================

if user_query:

    # --------------------------------------------------
    # Save User Message
    # --------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    st.chat_message(
        "user"
    ).write(
        user_query
    )


    # --------------------------------------------------
    # Assistant Response
    # --------------------------------------------------

    with st.chat_message("assistant"):

        callback = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=False,
        )


        # --------------------------------------------------
        # Prompt
        # --------------------------------------------------

        prompt = f"""
You are an expert SQL assistant.

Follow these rules carefully:

1. First inspect the database schema.
2. Use the available SQL database tools to answer the question.
3. Never invent tables or columns.
4. Generate SQL based only on the actual database schema.
5. Execute the SQL query before giving the final answer.
6. Do not guess the answer.
7. Return a clear and concise final answer.

User Question:

{user_query}
"""


        # --------------------------------------------------
        # Invoke Agent
        # --------------------------------------------------

        try:

            response = agent.invoke(
                {
                    "input": prompt
                },
                config={
                    "callbacks": [callback]
                },
            )


            # --------------------------------------------------
            # Get Final Answer
            # --------------------------------------------------

            answer = response["output"]


            # --------------------------------------------------
            # Display Answer
            # --------------------------------------------------

            st.write(answer)


            # --------------------------------------------------
            # Save Assistant Message
            # --------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


        except Exception as e:

            st.error(
                f"Error while processing your question:\n\n{e}"
            )