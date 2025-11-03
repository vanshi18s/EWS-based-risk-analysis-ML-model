import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import requests
import json


def calculate_features(df):
    df['return'] = df['close'].pct_change()
    window = 5
    df['volatility'] = df['return'].rolling(window).std() * np.sqrt(window)
    risk_free_rate = 0.01 / 252
    mean_return = df['return'].rolling(window).mean()
    stdev_return = df['return'].rolling(window).std()
    df['sharpe'] = (mean_return - risk_free_rate) / stdev_return
    confidence_level = 0.95
    z = norm.ppf(1 - confidence_level)
    df['var'] = -(z * stdev_return * df['close'])
    future_window = 5
    risk_threshold = -0.05
    df['future_return'] = df['close'].shift(-future_window) / df['close'] - 1
    df['risk_label'] = (df['future_return'] < risk_threshold).astype(int)
    df.dropna(inplace=True)
    return df


def train_model_rf(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=104,
        class_weight='balanced',
        max_depth=8,
        min_samples_split=5
    )
    model.fit(X_resampled, y_resampled)
    return model


def plot_metrics(df, var_threshold, vol_threshold, y_test=None, test_probs=None):
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), constrained_layout=True)
    axs[0].plot(df.index, df['volatility'], label='Volatility')
    axs[0].axhline(vol_threshold, color='r', linestyle='--', label='Volatility Threshold (from Train Set)')
    axs[0].set_title('Rolling Volatility')
    axs[0].legend()

    axs[1].plot(df.index, df['sharpe'], label='Sharpe Ratio')
    axs[1].set_title('Rolling Sharpe Ratio')
    axs[1].legend()

    axs[2].plot(df.index, df['var'], label='Value at Risk (VaR)')
    axs[2].axhline(var_threshold, color='r', linestyle='--', label='VaR Threshold (from Train Set)')
    axs[2].set_title('Value at Risk (VaR)')
    axs[2].legend()

    if y_test is not None and test_probs is not None:
        axs[3].plot(y_test.index, test_probs, label='Predicted Risk Probability')
        axs[3].plot(y_test.index, y_test, 'ro', label='Actual Risk Event (Price Drop)', alpha=0.5)
        axs[3].set_title('Predicted Risk Probability vs Actual Risk')
        axs[3].legend()
    else:
        axs[3].text(0.5, 0.5, 'Model predictions will appear here after training.', horizontalalignment='center', verticalalignment='center')
        axs[3].set_axis_off()
    return fig


def get_gemini_sentiment(company_name, api_key):
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    search_query = f"Find the latest news and financial headlines for the company '{company_name}'."
    search_system_prompt = (
        "You are a financial news aggregator. Your task is to find the latest, most relevant news "
        "headlines for the given company using the Google Search tool and present them as a concise summary."
    )
    search_payload = {
        "contents": [{"parts": [{"text": search_query}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {
            "parts": [{"text": search_system_prompt}]
        }
    }
    try:
        search_response = requests.post(api_url, json=search_payload, timeout=120)
        if search_response.status_code != 200:
            return {"error": f"API Search Error: {search_response.status_code} - {search_response.text}"}
        search_result = search_response.json()
        if 'candidates' not in search_result or not search_result['candidates']:
             return {"error": "Invalid search response from API. No candidates."}
        news_context = search_result['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        return {"error": f"Network Error during search: {e}"}
    except Exception as e:
        return {"error": f"An unknown error occurred during search: {e}"}

    analyze_query = (
        f"Here is the latest news context for '{company_name}':\n\n{news_context}\n\n"
        f"Based *only* on this provided news context, please provide a financial risk analysis."
    )
    analyze_system_prompt = (
        "You are a world-class financial analyst. Your task is to analyze the provided news summary "
        "and provide a brief summary and a risk assessment. "
        "You must output your analysis in a structured JSON format."
    )
    json_schema = {
        "type": "OBJECT",
        "properties": {
            "risk_score": {
                "type": "NUMBER",
                "description": "A score from 0 (very low risk) to 10 (very high risk) based on the news sentiment."
            },
            "sentiment": {
                "type": "STRING",
                "description": "The overall sentiment of the news ('Positive', 'Negative', or 'Neutral')."
            },
            "summary": {
                "type": "STRING",
                "description": "A 2-3 sentence summary explaining the news and the reason for the given risk score."
            }
        },
        "required": ["risk_score", "sentiment", "summary"]
    }
    analyze_payload = {
        "contents": [{"parts": [{"text": analyze_query}]}],
        "systemInstruction": {
            "parts": [{"text": analyze_system_prompt}]
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": json_schema
        }
    }
    try:
        analyze_response = requests.post(api_url, json=analyze_payload, timeout=120)
        if analyze_response.status_code == 200:
            analyze_result = analyze_response.json()
            if 'candidates' not in analyze_result or not analyze_result['candidates']:
                return {"error": "Invalid analysis response from API. No candidates."}
            text_part = analyze_result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(text_part)
        else:
            return {"error": f"API Analysis Error: {analyze_response.status_code} - {analyze_response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Network Error during analysis: {e}"}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON response: {e}"}
    except Exception as e:
        return {"error": f"An unknown error occurred during analysis: {e}"}


# --- Streamlit App ---
st.title('Stock Risk Early Warning System Dashboard')
st.markdown("Using **Random Forest** (Technical) and **Gemini API** (News Sentiment) to assess risk.")

gemini_api_key = st.text_input("Enter your Gemini API Key", type="password")
uploaded_file = st.file_uploader("Upload your stock CSV file", type=["csv"])
required_columns = {'open', 'high', 'low', 'close', 'volume'}

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.lower()

        if not required_columns.issubset(df.columns):
            st.error(f"The uploaded CSV is missing required columns. It must have: {required_columns - set(df.columns)}")
        else:
            st.success("CSV file loaded. All required data columns found.")

            company_name = st.text_input("Enter Company Name (e.g., 'Apple', 'NVIDIA') for News Analysis")

            st.subheader("1. Select Your Date Column")
            potential_date_cols = ['date', 'time', 'timestamp', 'datetime']
            date_col_guess = next((col for col in potential_date_cols if col in df.columns), None)
            if date_col_guess is None:
                date_col_guess = df.columns[0]
            default_index = df.columns.tolist().index(date_col_guess)
            date_column_name = st.selectbox("Which column in your file contains the dates?", df.columns, index=default_index)

            st.subheader("2. Data Processing and Preview")
            with st.spinner(f"Processing date column '{date_column_name}'..."):
                df['date_parsed'] = pd.to_datetime(df[date_column_name], errors='coerce')
                if df['date_parsed'].isnull().any():
                    st.error(f"Error: Could not parse all dates in the column '{date_column_name}'. Please check the CSV file or select a different column.")
                else:
                    df.set_index('date_parsed', inplace=True)
                    df = df.sort_index()
                    st.success("Date column parsed and set as index successfully!")

                    df = calculate_features(df)
                    st.dataframe(df.tail())

                    train_size = int(len(df) * 0.7)
                    val_size = int(len(df) * 0.15)

                    features_df = df[['volatility', 'sharpe', 'var', 'return']]
                    target_series = df['risk_label']

                    X_train = features_df.iloc[:train_size]
                    y_train = target_series.iloc[:train_size]
                    X_val = features_df.iloc[train_size:train_size + val_size]
                    y_val = target_series.iloc[train_size:train_size + val_size]
                    X_test = features_df.iloc[train_size + val_size:]
                    y_test = target_series.iloc[train_size + val_size:]

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    X_test_scaled = scaler.transform(X_test)

                    var_threshold = X_train['var'].quantile(0.75)
                    vol_threshold = X_train['volatility'].quantile(0.75)

                    st.subheader("3. Technical Risk Model (Random Forest)")
                    st.write(f"Training on {len(X_train)} samples, validating on {len(X_val)}, testing on {len(X_test)}.")
                    st.write(f"**Positive Class (Risk Event):** {y_train.sum()} events in training set ({y_train.mean():.2%}).")

                    if y_train.sum() < 5:
                        st.warning("Warning: Very few risk events detected in the training set. The technical model may struggle to learn.")

                    if st.button('Train Technical Risk Model'):
                        with st.spinner('Training Random Forest model...'):
                            model = train_model_rf(X_train_scaled, y_train)

                            test_probs = model.predict_proba(X_test_scaled)[:, 1]
                            y_pred_test = model.predict(X_test_scaled)

                            accuracy = accuracy_score(y_test, y_pred_test)
                            st.subheader(f"Technical Model Accuracy on Test Set: {accuracy:.2%}")
                            st.info("This model predicts future price drops based on historical technical data.")

                            # Classification report intentionally NOT displayed here

                            fig = plot_metrics(df, var_threshold, vol_threshold, y_test, test_probs)
                            st.pyplot(fig)

                            risk_df = df.iloc[train_size + val_size:].copy()
                            risk_df['Risk_Probability'] = test_probs
                            risk_df_csv = risk_df.to_csv(index=True)

                            st.download_button(
                                label="Download Technical Risk CSV",
                                data=risk_df_csv,
                                file_name='technical_risk_probabilities.csv',
                                mime='text/csv'
                            )

                            high_risk_days = risk_df[risk_df['Risk_Probability'] > 0.7]
                            if not high_risk_days.empty:
                                st.warning(f'Technical model high risk alert! {len(high_risk_days)} days with predicted risk probability > 0.7 detected.')
                                st.dataframe(high_risk_days[['Risk_Probability', 'future_return']])
                            else:
                                st.success('Technical model found no high-probability risk days (> 0.7) in test set.')

            st.subheader("4. News & Sentiment Risk Analysis (Gemini)")
            if company_name and gemini_api_key:
                if st.button(f"Analyze News & Sentiment for {company_name}"):
                    with st.spinner(f"Gemini is analyzing the latest news for {company_name}..."):
                        sentiment_data = get_gemini_sentiment(company_name, gemini_api_key)
                        if "error" in sentiment_data:
                            st.error(f"Error from Gemini API: {sentiment_data['error']}")
                        else:
                            st.subheader(f"Gemini Risk Analysis for {company_name}")
                            score = sentiment_data.get('risk_score', 0)
                            st.metric("Gemini Risk Score (0-10)", f"{score}/10")

                            sentiment = sentiment_data.get('sentiment', 'N/A')
                            if sentiment == "Positive":
                                st.success(f"Sentiment: **{sentiment}**")
                            elif sentiment == "Negative":
                                st.error(f"Sentiment: **{sentiment}**")
                            else:
                                st.info(f"Sentiment: **{sentiment}**")

                            st.write("---")
            else:
                st.info("Enter a Company Name and your Gemini API Key above to enable real-time news analysis.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.exception(e)
else:
    st.info("Upload a CSV file and enter your Gemini API Key to start analysis.")
