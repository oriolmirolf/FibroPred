# app.py

import streamlit as st

def main():
    st.set_page_config(page_title="FibroPred CBR Application", layout='wide')
    st.title("FibroPred CCB Application")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Train Model", "Predict"])

    if page == "Train Model":
        from pages import train_model
        train_model.run()
    elif page == "Predict":
        from pages import predict
        predict.run()

if __name__ == '__main__':
    main()
