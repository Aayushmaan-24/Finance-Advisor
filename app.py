import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Streamlit UI
st.set_page_config(page_title="💰 AI-Powered Finance Advisor", layout="wide")
st.title("💸 Smart Finance Planner: AI-Powered Insights")
st.markdown("### Plan, Predict & Grow Your Wealth! 🚀")

# Sidebar for User Inputs
st.sidebar.header("🔹 Financial Inputs")

# Currency Selection
currency = st.sidebar.selectbox("Select Currency:", ["₹", "$", "€", "£", "¥"], index=0)

# User Inputs
income = st.sidebar.number_input(f"Monthly Income ({currency}):", min_value=0)
expenses = st.sidebar.number_input(f"Monthly Expenses ({currency}):", min_value=0)
savings = income - expenses

goal = st.sidebar.number_input(f"Savings Goal ({currency}):", min_value=0)
time_period = st.sidebar.number_input("Target Months to Achieve Goal:", min_value=1)

# Analysis Section
st.subheader("📊 Financial Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("💰 Monthly Savings", f"{currency}{savings}")
    if savings > 0:
        months_needed = goal / savings if savings > 0 else float('inf')
        st.metric("🎯 Time to Achieve Goal", f"{months_needed:.1f} months")
    else:
        st.error("⚠️ Your expenses exceed income! Consider adjusting your budget.")

with col2:
    labels = ['Expenses', 'Savings']
    values = [expenses, savings if savings > 0 else 0]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=['#FF6F61', '#6B8E23'])
    ax.set_title("💡 Savings vs Expenses")
    st.pyplot(fig)

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
initial_investment = st.number_input(f"Initial Investment Amount ({currency}):", min_value=0)
monthly_contribution = st.number_input(f"Monthly Investment Contribution ({currency}):", min_value=0)
investment_years = st.slider("Investment Duration (Years):", 1, 30, 10)
expected_return = {'Low': 5, 'Medium': 10, 'High': 15}[option]  # Expected annual return in %

# Monte Carlo Simulation for Investment Growth
def monte_carlo_simulation(initial_investment, monthly_contribution, years, expected_return, num_simulations=1000):
    simulations = []
    for _ in range(num_simulations):
        investment_value = initial_investment
        values = [investment_value]
        for _ in range(years * 12):
            monthly_growth = np.random.normal(expected_return / 12 / 100, 0.05)
            investment_value = investment_value * (1 + monthly_growth) + monthly_contribution
            values.append(investment_value)
        simulations.append(values)
    return np.array(simulations)

# Run Monte Carlo Simulation
simulations = monte_carlo_simulation(initial_investment, monthly_contribution, investment_years, expected_return)

# Plot the simulations
fig3, ax3 = plt.subplots()
ax3.plot(simulations.T, color='lightblue', alpha=0.5)
ax3.set_title("Monte Carlo Simulation: Investment Growth Over Time")
ax3.set_xlabel("Months")
ax3.set_ylabel(f"Investment Value ({currency})")
st.pyplot(fig3)

# Real-time Market Data: Stocks & Crypto
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
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name=f"{stock_ticker} Closing Price"))
            fig.update_layout(
                title=f"📊 {stock_ticker} Stock Price Trend",
                xaxis_title="Date",
                yaxis_title=f"Price ({currency})",
                template="plotly_dark"
            )
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"⚠️ Error fetching data: {e}. Please try again with a valid stock symbol.")
        st.info("Example stock symbols: AAPL (Apple), TSLA (Tesla), BTC-USD (Bitcoin), AMZN (Amazon)")

# Personalized Recommendations
def personalized_recommendations(income, savings, risk_level, goal):
    if savings < goal * 0.25:
        return "💡 You need to increase your savings rate to meet your goal on time."
    if risk_level == "Low":
        return "🔒 Consider safer investments like bonds, fixed deposits, or index funds."
    if risk_level == "Medium":
        return "📊 A balanced portfolio with stocks, ETFs, and mutual funds might suit you."
    if risk_level == "High":
        return "🚀 Look into riskier assets like crypto, startups, or individual stocks."

recommendation = personalized_recommendations(income, savings, option, goal)
st.info(recommendation)

# Expense Tracker
expense_categories = ["Rent", "Groceries", "Entertainment", "Utilities", "Others"]
expense_values = {}

for category in expense_categories:
    expense_values[category] = st.sidebar.number_input(f"{category} ({currency}):", min_value=0)

total_expenses = sum(expense_values.values())

# Pie chart to visualize expenses
fig_expenses, ax_expenses = plt.subplots()
ax_expenses.pie(expense_values.values(), labels=expense_values.keys(), autopct='%1.1f%%', colors=sns.color_palette("Set3", len(expense_values)))
ax_expenses.set_title("🧾 Expense Breakdown")
st.pyplot(fig_expenses)

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
