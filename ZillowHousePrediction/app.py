import streamlit as st
import pandas as pd
import joblib
import openai
import os

# Load model  - already trained and saved in the same directory
model = joblib.load("model.pkl")

# Set OpenAI API key  - loading on rutime from command line
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client()

st.set_page_config(page_title="Home Price Predictor", layout="centered")
st.title("Home Price Prediction with GPT-4")
st.markdown("Enter property details and get an instant price prediction with an AI explanation.")

# Input form
with st.form("form"):
    bed = st.slider("Bedrooms", 1, 10, 3)
    bath = st.slider("Bathrooms", 1, 10, 2)
    acre_lot = st.number_input("Acre Lot", value=0.15)
    house_size = st.number_input("House Size (sq ft)", value=1500)
    zip_code = st.text_input("ZIP Code", "10001")
    submit = st.form_submit_button("Predict Price")

if submit:
    try:
        zip_code = float(zip_code)
        input_df = pd.DataFrame([{
            'bed': bed,
            'bath': bath,
            'acre_lot': acre_lot,
            'house_size': house_size,
            'zip_code': zip_code
        }])
        
        predicted_price = model.predict(input_df)[0]
        st.success(f"💰 Predicted Price: ${predicted_price:,.2f}")

        # GPT explanation
        prompt = f"""
        A house has {bed} bedrooms, {bath} bathrooms, a lot size of {acre_lot} acres,
        a house size of {house_size} square feet, and is located in ZIP code {zip_code}.
        The predicted price is ${predicted_price:,.2f}.
        Explain this in simple terms and suggest what could increase the price.
        """

        with st.spinner("Getting explanation from GPT-4..."):
            response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful real estate assistant."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=500,
    temperature=0.7
)
            explanation = response.choices[0].message.content
            st.markdown("### 🤖 GPT-4 Insight")
            st.write(explanation)

    except Exception as e:
        st.error(f"Error: {e}")
