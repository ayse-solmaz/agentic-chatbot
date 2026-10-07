from agentic_chatbot_db_backend import chatbot, get_all_threads
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st
import uuid 

# generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())

# add a new thread ID to the conversation list
def add_thread(thread_id):

    # prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)

# create a completely new chat conversation
def reset_chat():

    # generate and assign a new thread ID
    st.session_state["thread_id"] = generate_thread_id()

    # clear the current chat messages from the UI
    st.session_state["message_history"] = []

    # add the new thread to the conversation list
    add_thread(st.session_state["thread_id"])

# load a previous conversation from the langgraph checkpointer
def load_conversation(thread_id):

    # get the saved state for the selected thread
    state = chatbot.get_state(
        config={
            "configurable":{
                "thread_id":thread_id
            }
        }
    )

    # return saved messages
    # return an empty list if no messages are available
    return state.values.get("messages",[])

# display the main application title
st.title("agentic chatbot with langgraph")

# create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()


# add the current thread to the conversation list
add_thread(st.session_state["thread_id"])

# Display the sidebar title
st.sidebar.title("My Conversations")


# create a button for starting a new conversation
if st.sidebar.button("new chat"):

    # reset the current chat and create a new thread
    reset_chat()

    # rerun the streamlit app to update the interface
    st.rerun()

# display all conversation threads in reverse order
# this shows the newest conversation first

for thread_id in st.session_state["chat_threads"][::-1]:

    # create one sidebar button for every conversation
    if st.sidebar.button(str(thread_id),key=thread_id):

        # set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # temporary list for converting langchain messages
        # into streamlit's required message format

        temp_messages = []

        # loop through all saved messages
        for message in messages:

            # check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"
            # check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"
            # ignore other message types, such as ToolMessage
            else:
                continue

            # convert the langchain message into a dictionary
            temp_messages.append({"role": role, "content": message.content})

        # replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # rerun the application to display the loaded messages
        st.rerun()


# display all messages from the  currently selected conversation
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("type here")

if user_input:
    st.session_state["message_history"].append({"role":"user","content":user_input})

    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable":{"thread_id":st.session_state["thread_id"]},
        "metadata":{"thread_id":st.session_state["thread_id"]},
        "run_name":"chat_trace",
    }

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            message_chunk.content

            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            )
            if isinstance(message_chunk, AIMessage)
        )

    # save the assistant's response in session state
    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})