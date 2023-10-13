from langchain.agents import ConversationalChatAgent, AgentExecutor  # Importing the necessary classes for the chat agent and executor
from langchain.callbacks import StreamlitCallbackHandler  # Importing the callback handler for Streamlit
from langchain.chat_models import ChatOpenAI  # Importing the chat model for OpenAI
from langchain.memory import ConversationBufferMemory  # Importing the memory class for conversation buffer
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory  # Importing the chat message history class for Streamlit
from langchain.tools import DuckDuckGoSearchRun  # Importing the tool for DuckDuckGo search
import streamlit as st  # Importing the Streamlit library for building the web application

st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦜")  # Setting the page configuration for the Streamlit app
st.title("Hey, I am you new colleague. Do you have any research task? I am here to help")  # Adding a title to the Streamlit app

#openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")  # Creating a text input field in the sidebar for the OpenAI API key
st.write(header.jpeg)

msgs = StreamlitChatMessageHistory()  # Creating an instance of the StreamlitChatMessageHistory class for storing chat messages
memory = ConversationBufferMemory(  # Creating an instance of the ConversationBufferMemory class for storing conversation history
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):  # Checking if the chat history is empty or if the reset button is clicked
    msgs.clear()  # Clearing the chat history
    msgs.add_ai_message("How can I help you?")  # Adding an AI message to start the conversation
    st.session_state.steps = {}  # Initializing the session state for storing intermediate steps

avatars = {"human": "user", "ai": "assistant"}  # Mapping the avatars for different message types
for idx, msg in enumerate(msgs.messages):  # Looping through the chat messages
    with st.chat_message(avatars[msg.type]):  # Displaying the chat message with the corresponding avatar
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):  # Looping through the intermediate steps for the current message
            if step[0].tool == "_Exception":  # Skipping the step if it is an exception
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):  # Displaying the tool name and input as a status message
                st.write(step[0].log)  # Displaying the log of the tool execution
                st.write(step[1])  # Displaying the output of the tool
        st.write(msg.content)  # Displaying the content of the message

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):  # Creating a chat input field for the user to enter a prompt
    st.chat_message("user").write(prompt)  # Adding the user's prompt as a chat message

    if not openai_api_key:  # Checking if the OpenAI API key is empty
        st.info("Please add your OpenAI API key to continue.")  # Displaying an info message to add the API key
        st.stop()  # Stopping the execution of the Streamlit app

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)  # Creating an instance of the ChatOpenAI class for the chat model
    tools = [DuckDuckGoSearchRun(name="Search")]  # Creating a list of tools with the DuckDuckGo search tool
    chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=llm, tools=tools)  # Creating an instance of the ConversationalChatAgent class with the chat model and tools
    executor = AgentExecutor.from_agent_and_tools(  # Creating an instance of the AgentExecutor class with the agent, tools, and memory
        agent=chat_agent,
        tools=tools,
        memory=memory,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):  # Displaying the chat message with the assistant avatar
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)  # Creating an instance of the StreamlitCallbackHandler class
        response = executor(prompt, callbacks=[st_cb])  # Executing the chat agent with the prompt and callback handler
        st.write(response["output"])  # Displaying the output of the chat agent
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]  # Storing the intermediate steps in the session state


