import streamlit as st
from utils import speech_to_text, text_to_speech, query_gemini, update_history, clean_text, play_audio_streamed, delete_history


st.title("ChatMate.ai - Your English Speaking Teacher and Buddy 🎙️")
st.markdown("**Speak to ChatMate, and it'll guide you to improve your English!**")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Enter your name and choose a topic to start chatting.
2. Click on 'Record' to start speaking.
3. The app converts your speech to text and responds based on your chosen topic.
4. Say 'end chat' or click 'End Chat' to finish the session.
""")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "chat_topic" not in st.session_state:
    st.session_state.chat_topic = None

if "chat_active" not in st.session_state:
    st.session_state.chat_active = False

#--> Get User Name and Topic
if not st.session_state.user_name:
    st.session_state.user_name = st.text_input("Enter your name to start chatting:", key="user_name_input")
if st.session_state.user_name and not st.session_state.chat_topic:
    st.session_state.chat_topic = st.text_input(
        f"Hi {st.session_state.user_name}, what topic would you like to discuss?", key="chat_topic_input"
    )
    if st.session_state.chat_topic:
        update_history({
                        "role": "user",
                        "parts": f"The topic of discussion is {st.session_state.chat_topic}"
        })
        update_history({
                        "role": "model",
                        "parts": f"Oh {st.session_state.chat_topic} is nice topic of discussion."
        })


#--> Start Chat Button
if st.session_state.user_name and st.session_state.chat_topic and not st.session_state.chat_active:
    if st.button("Start Chat"):
        st.session_state.chat_active = True
        st.success(f"Chat started! Topic: {st.session_state.chat_topic}")

st.subheader("Chat with ChatMate.ai")

if st.session_state.chat_active:
    if st.button("Record"):
        user_input = speech_to_text()
        if user_input:
            print(user_input)
            if "exit" in user_input.lower():  # End chat trigger
                st.session_state.chat_active = False
                delete_history()
                st.success("Chat ended. Thank you for using ChatMate.ai!")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                llm_response = query_gemini(user_input)
                update_history({
                    "role":"user",
                    "parts":user_input
                })
                update_history({
                    "role": "model",
                    "parts": llm_response
                })
                st.session_state.chat_history.append({"role": "assistant", "content": llm_response})
                st.markdown(f"{st.session_state.user_name}: {user_input}")
                st.success(f"ChatMate's Response: {llm_response}")
                play_audio_streamed(clean_text(llm_response))
                

    # Display Chat History
    if st.button("Chat History"):
        st.write("### Chat History")
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"**{st.session_state.user_name}:** <br>{chat['content']}", unsafe_allow_html=True)
            elif chat["role"] == "assistant":
                st.markdown(f"**ChatMate.ai:**\n{chat['content']}")

if st.session_state.chat_active and st.button("End Chat"):
    st.session_state.chat_active = False
    delete_history()
    st.session_state.user_name = None
    st.session_state.chat_topic = None
    st.success("Chat ended. Thank you for using ChatMate.ai!")