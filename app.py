import streamlit as st
from utils import speech_to_text, text_to_speech, query_gemini, gen_prompt, play_audio

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

#--> Start Chat Button
if st.session_state.user_name and st.session_state.chat_topic and not st.session_state.chat_active:
    if st.button("Start Chat"):
        st.session_state.chat_active = True
        st.success(f"Chat started! Topic: {st.session_state.chat_topic}")
        role="You are an English speaking teacher. You will help me learn spoken english by asking me on what topic should we start our conversation."
        prompt=gen_prompt(role)
        reply=f"My name is {st.session_state.user_name}. I want to learn english. The topic of discussion is {st.session_state.chat_topic}"
        prompt=gen_prompt(reply)

st.subheader("Chat with ChatMate.ai")

if st.session_state.chat_active:
    if st.button("Record"):
        user_input = speech_to_text()
        if user_input:
            print(user_input)
            if "exit" in user_input.lower():  # End chat trigger
                st.session_state.chat_active = False
                st.success("Chat ended. Thank you for using ChatMate.ai!")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                prompt=gen_prompt(user_input)
                llm_response = query_gemini(prompt)
                st.session_state.chat_history.append({"role": "assistant", "content": llm_response})
                st.success("ChatMate's Response: " + llm_response)

                audio_file = text_to_speech(llm_response)
                st.audio(audio_file, format="audio/mp3", autoplay=True)

    # Display Chat History
    st.write("### Chat History")
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**{st.session_state.user_name}:** {chat['content']}")
        elif chat["role"] == "assistant":
            st.markdown(f"**ChatMate:** {chat['content']}")

if st.session_state.chat_active and st.button("End Chat"):
    st.session_state.chat_active = False
    st.session_state.user_name = None
    st.session_state.chat_topic = None
    st.success("Chat ended. Thank you for using ChatMate.ai!")