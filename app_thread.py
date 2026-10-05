from agentic_chatbot_backend import chatbot 
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
        config = {"configurable":{"thread_id":thread_id}}
    )
    # return saved messages
    # return an emopty list if no messages are available
    return state.values.get("messages",[])

# display the main application title
st.title("agentic chatbot with langgraph")

# create messsage_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

# create a list of storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

# add the current thread to the conversation list
add_thread(st.session_state["thread_id"])

# display the sidebar title
st.sidebar.title("my conversations")

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
    if st.sidebar.button(
        str(thread_id),
        key=thread_id
    ):
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
            if isinstance(message,HumanMessage):
                role = "user"
            # check whether the message was sent by the AI
            elif isinstance(message,AIMessage):
                role = "assistant"
            # ignore other message types
            else:
                continue

            # convert the langchain message into a dictionary
            temp_messages.append({"role":role,"content":message.content})

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()

        # Display all messages from the currently selected conversation
        
for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content
        st.text(message["content"])



# Create the chat input box
user_input = st.chat_input("type here")


# Run this block after the user submits a message
if user_input:

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })


    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)


    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }


    # Create the assistant chat-message container
    with st.chat_message("assistant"):

        # Stream the assistant response token by token
        ai_message = st.write_stream(

            # Return only the content of AI message chunks
            message_chunk.content

            # stream messages from the LangGraph chatbot
            for message_chunk, metadata in chatbot.stream(
                {
                    # send the latest user message to the chatbot
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },

                # use the current conversation thread
                config=CONFIG,

                # stream individual message chunks
                stream_mode="messages"
            )

            # display only AI messages
            # this prevents tool and user messages from appearing
            if isinstance(message_chunk, AIMessage)
        )


    # save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })