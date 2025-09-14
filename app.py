# app.py
import streamlit as st
import pandas as pd
from classifier import classify_ticket, TOPICS
import rag
import os

st.set_page_config(page_title="Customer Support Copilot", layout="wide")
st.title("Customer Support Copilot — Demo")

# Sidebar
st.sidebar.header("Setup & KB")
st.sidebar.write("Build or re-build the knowledge index used for RAG.")
if st.sidebar.button("Build KB index"):
    try:
        with st.spinner("Building index (this may take a while)..."):
            rag.build_index(kb_dir="kb")
        st.sidebar.success("Index built successfully.")
    except Exception as e:
        st.sidebar.error(f"Index build failed: {e}")

st.sidebar.markdown("---")
st.sidebar.write("OpenAI API Key (optional for local testing). If left empty, app will try local model.")
api_key_input = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

# Load or upload tickets
st.header("Bulk ticket classification")
uploaded = st.file_uploader("Upload sample_tickets CSV (columns: id,channel,subject,text)", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if os.path.exists("sample_tickets.csv"):
        df = pd.read_csv("sample_tickets.csv")
    else:
        df = pd.DataFrame(columns=["id","channel","subject","text"])

if df.empty:
    st.info("No tickets loaded. Upload sample_tickets.csv or run create_sample_tickets.py.")
else:
    if "classified_df" not in st.session_state:
        with st.spinner("Classifying tickets..."):
            rows = []
            for _, r in df.iterrows():
                cl = classify_ticket(r.get("text",""))
                row = {**r.to_dict(), **cl}
                rows.append(row)
            st.session_state["classified_df"] = pd.DataFrame(rows)
    st.dataframe(st.session_state["classified_df"], use_container_width=True)
    st.download_button("Download classifications (.csv)", st.session_state["classified_df"].to_csv(index=False), file_name="classified_tickets.csv")

# Interactive agent
st.header("Interactive AI Agent")
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("Submit a new ticket")
    channel = st.selectbox("Channel", ["Email","Chat","WhatsApp","Voice"])
    subject = st.text_input("Subject")
    text = st.text_area("Ticket content", height=200)
    if st.button("Submit Ticket"):
        if not text.strip():
            st.warning("Enter ticket text first.")
        else:
            analysis = classify_ticket(text)
            st.session_state["last_ticket"] = {"channel": channel, "subject": subject, "text": text, "analysis": analysis}
            st.success("Ticket analyzed — see the Internal Analysis and Final Response panes.")

with col2:
    st.subheader("Internal Analysis (Back-end view)")
    if "last_ticket" in st.session_state:
        st.json(st.session_state["last_ticket"]["analysis"])
    else:
        st.write("Submit a ticket to see internal analysis.")

st.subheader("Final Response (Front-end view)")
if "last_ticket" in st.session_state:
    analysis = st.session_state["last_ticket"]["analysis"]
    tags = analysis["tags"]
    eligible = set(["How-to","Product","Best practices","API/SDK","SSO"])
    if any(t in eligible for t in tags):
        st.info("This topic is RAG-eligible — generating an answer using the KB.")
        try:
            retrieved = rag.retrieve(st.session_state["last_ticket"]["text"])
            result = rag.answer_with_rag(st.session_state["last_ticket"]["text"], retrieved)
            st.markdown(f"**AI Answer (Model: {result['model']}):**")
            st.write(result["answer"])
            st.markdown("**Sources used:**")
            for s in result["sources"]:
                st.write(s)
        except Exception as e:
            st.error(f"RAG failed: {e}. Make sure index is built or KB contains data.")
    else:
        st.info(f"This ticket has been classified as a '{tags[0]}' issue and routed to the appropriate team.")
else:
    st.write("Submit a ticket to see the final response.")
