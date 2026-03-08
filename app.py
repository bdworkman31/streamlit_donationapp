import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model artifacts
model = joblib.load("donation-model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Build campaign dropdown from model columns
campaign_cols = [c for c in model_columns if c.startswith("campaign_")]


st.title("Donor Prediction Model")
st.write("Enter the info below to determine if the person will be a high or low donor")

#Dropdown box for campaign values obtained from OHE campaign column values to match up with deployed model

campaign_options = [col.replace("campaign_", "") for col in campaign_cols]



#User inputs 

comment = st.text_area("Comment")
credit_card = st.selectbox("Was a credit card used?", ["No", "Yes"])
campaign = st.selectbox("Campaign", campaign_options)
donation_date = st.date_input("Donation Date")


#Create Preprocessing function that can run in the app 

def preprocess_df(comment, credit_card, campaign, donation_date):
    comment_word_length = len(str(comment).split()) if comment else 0
    likely_cc = 1 if credit_card == "Yes" else 0

    donation_date = pd.to_datetime(donation_date)
    year = donation_date.year
    month = donation_date.month
    day = donation_date.day
    day_of_week = donation_date.dayofweek

    # Create one-row dataframe with all model columns set to 0
    processed = pd.DataFrame(0, index=[0], columns=model_columns)

    # Fill numeric features
    processed.at[0, "likely_cc"] = likely_cc
    processed.at[0, "day"] = day
    processed.at[0, "year"] = year
    processed.at[0, "day_of_week"] = day_of_week
    processed.at[0, "month"] = month
    processed.at[0, "comment_word_length"] = comment_word_length

    # Turn on selected campaign column (set to 1)
    campaign_col = f"campaign_{campaign}"
    if campaign_col in processed.columns:
        processed.at[0, campaign_col] = 1

    return processed


if st.button("Predict"):
    processed_input = preprocess_df(
        comment,
        credit_card,
        campaign,
        donation_date
    )

    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    if prediction == 1:
        st.success("Yes")
    else:
        st.error("No")

    st.write(f"Probability of high donation: {probability:.2%}")
