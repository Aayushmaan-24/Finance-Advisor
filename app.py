import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import yfinance as yf
import plotly.express as px

# Streamlit UI
st.set_page_config(page_title="💰 AI-Powered Finance Advisor", layout="wide")
st.title("💸 Smart Finance Planner: AI-Powered Insights")
st.markdown("### Plan, Predict & Grow Your Wealth! 🚀")

# Sidebar for User Inputs
st.sidebar.header("🔹 Financial Inputs")

# Currency Selection
currency = st.sidebar.selectbox("Select Currency:", ["₹", "$", "€", "£", "¥"], index=0)

# User Inputs: Income
income = st.sidebar.number_input(f"Monthly Income ({currency}):", min_value=0.0, step=100.0)

# Expense Categories
st.sidebar.subheader("💸 Monthly Expenses")
expense_categories = {
    "Rent/Mortgage": st.sidebar.number_input(f"Rent/Mortgage ({currency}):", min_value=0.0, step=10.0),
    "Groceries": st.sidebar.number_input(f"Groceries ({currency}):", min_value=0.0, step=10.0),
    "Utilities": st.sidebar.number_input(f"Utilities ({currency}):", min_value=0.0, step=10.0),
    "Transportation": st.sidebar.number_input(f"Transportation ({currency}):", min_value=0.0, step=10.0),
    "Entertainment": st.sidebar.number_input(f"Entertainment ({currency}):", min_value=0.0, step=10.0),
}

# Allow custom expense categories
st.sidebar.subheader("➕ Add Custom Expense")
custom_expense_name = st.sidebar.text_input("Custom Expense Name (e.g., Gym):")
custom_expense_amount = st.sidebar.number_input(f"Custom Expense Amount ({currency}):", min_value=0.0, step=10.0)
if custom_expense_name and custom_expense_amount > 0:
    expense_categories[custom_expense_name] = custom_expense_amount

# Calculate total expenses and savings
total_expenses = sum(expense_categories.values())
savings = income - total_expenses

# Savings Goal and Time Period
goal = st.sidebar.number_input(f"Savings Goal ({currency}):", min_value=0.0, step=100.0)
time_period = st.sidebar.number_input("Target Months to Achieve Goal:", min_value=1, step=1)

# Analysis Section
st.subheader("📊 Financial Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("💰 Monthly Savings", f"{currency}{savings:.2f}")
    if savings > 0:
        months_needed = goal / savings if savings > 0 else float('inf')
        st.metric("🎯 Time to Achieve Goal", f"{months_needed:.1f} months")
    else:
        st.error("⚠️ Your expenses exceed income! Consider adjusting your budget.")

with col2:
    # Prepare data for pie chart
    labels = list(expense_categories.keys()) + ["Savings"]
    values = list(expense_categories.values()) + [savings if savings > 0 else 0]
    # Filter out zero values to avoid clutter
    filtered_labels = [label for label, value in zip(labels, values) if value > 0]
    filtered_values = [value for value in values if value > 0]
    
    if sum(filtered_values) > 0:  # Ensure total is not zero
        fig, ax = plt.subplots()
        ax.pie(filtered_values, labels=filtered_labels, autopct='%1.1f%%', colors=sns.color_palette("Set2", len(filtered_labels)))
        ax.set_title("💡 Expenses Breakdown and Savings")
        st.pyplot(fig)
    else:
        st.warning("⚠️ No valid data to display in pie chart (all expenses and savings are zero).")

# Savings Trend Prediction
st.subheader("📈 Future Savings Prediction")
months = np.array(range(1, 13)).reshape(-1, 1)
savings_trend = np.array([savings * (1 + 0.02 * i) for i in range(12)])  # Simulating a 2% increase per month

model = LinearRegression()
model.fit(months, savings_trend)
future_months = np.array(range(1, 25)).reshape(-1, 1)
predicted_savings = model.predict(future_months)

fig2, ax2 = plt.subplots()
ax2.plot(future_months, predicted_savings, label="Projected Savings", color='blue')
ax2.set_xlabel("Months")
ax2.set_ylabel(f"Savings ({currency})")
ax2.set_title("🔮 Predicted Savings Growth")
ax2.legend()
st.pyplot(fig2)

# Investment Insights
st.subheader("💹 Investment Options Based on Risk Level")
option = st.radio("Choose Your Risk Tolerance:", ['Low', 'Medium', 'High'])

investment_strategies = {
    'Low': "✅ Fixed Deposits, Government Bonds, Index Funds",
    'Medium': "📊 Mutual Funds, REITs, Balanced Portfolio",
    'High': "🚀 Stocks, Crypto, Startups, Commodities"
}

st.info(f"For a **{option}-risk** strategy, consider: {investment_strategies[option]}")

# Investment Simulation with Real-time Market Data
st.subheader("📊 Investment Growth Simulation")
initial_investment = st.number_input(f"Initial Investment Amount ({currency}):", min_value=0.0, step=100.0)
monthly_contribution = st.number_input(f"Monthly Investment Contribution ({currency}):", min_value=0.0, step=10.0)
investment_years = st.slider("Investment Duration (Years):", 1, 30, 10)
expected_return = {'Low': 5, 'Medium': 10, 'High': 15}[option]  # Expected annual return in %

# Fetch real-time stock data
st.subheader("📈 Real-Time Market Data")
stock_ticker = st.text_input("Enter a stock symbol (e.g., AAPL, TSLA, BTC-USD):", "AAPL")

if stock_ticker:
    try:
        # Fetch real-time data using yfinance
        stock_data = yf.Ticker(stock_ticker).history(period="1y")
        
        if stock_data.empty:
            st.warning("⚠️ Invalid stock symbol or no data available. Please try a valid stock symbol.")
            st.info("Example stock symbols: AAPL (Apple), TSLA (Tesla), BTC-USD (Bitcoin), AMZN (Amazon)")
        else:
            # Plot stock data using Plotly
            fig4 = px.line(
                stock_data,
                x=stock_data.index,
                y='Close',
                labels={'x': 'Date', 'Close': f'Price ({currency})'},
                title=f"📊 {stock_ticker} Stock Price Trend"
            )
            fig4.update_traces(line_color='purple', name=f"{stock_ticker} Closing Price")
            fig4.update_layout(showlegend=True)
            st.plotly_chart(fig4, use_container_width=True)
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}. Please try again with a valid stock symbol.")
        st.info("Example stock symbols: AAPL (Apple), TSLA (Tesla), BTC-USD (Bitcoin), AMZN (Amazon)")

# Simulating Investment Growth
months = np.arange(1, investment_years * 12 + 1)
investment_values = np.zeros(len(months))
current_value = initial_investment
for i in range(len(months)):
    current_value *= (1 + expected_return / 100 / 12)
    current_value += monthly_contribution
    investment_values[i] = current_value

fig3, ax3 = plt.subplots()
ax3.plot(months, investment_values, label=f"{option} Risk Investment Growth", color='green')
ax3.set_xlabel("Months")
ax3.set_ylabel(f"Investment Value ({currency})")
ax3.set_title("📈 Simulated Investment Growth Over Time")
ax3.legend()
st.pyplot(fig3)

# Suggestions & Insights
st.subheader("🚀 AI-Powered Recommendations")
if savings > 0:
    if months_needed <= time_period:
        st.success("✅ You're on track to meet your savings goal! Keep it up! 🎉")
    else:
        extra_savings = (goal / time_period) - savings
        st.warning(f"💡 To achieve your goal faster, save an additional {currency}{extra_savings:.2f} per month.")
        st.info("Consider cutting discretionary spending or exploring investment options.")
else:
    st.error("⚠️ High expenses detected! Try reducing non-essential spending or increasing income streams.")