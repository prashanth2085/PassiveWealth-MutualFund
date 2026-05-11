import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import requests

# --- CUSTOM MATH & SIMULATION FUNCTIONS ---
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/window, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_cagr(start_value, end_value, years):
    if years <= 0 or start_value <= 0: return 0
    return ((end_value / start_value) ** (1 / years) - 1) * 100

def simulate_sip(price_series, monthly_investment):
    df = price_series.to_frame(name='NAV')
    df['YearMonth'] = df.index.to_period('M')
    monthly_data = df.groupby('YearMonth').first()
    
    monthly_data['Units_Bought'] = monthly_investment / monthly_data['NAV']
    monthly_data['Cumulative_Units'] = monthly_data['Units_Bought'].cumsum()
    monthly_data['Total_Invested'] = np.arange(1, len(monthly_data) + 1) * monthly_investment
    monthly_data['Portfolio_Value'] = monthly_data['Cumulative_Units'] * monthly_data['NAV']
    
    return monthly_data

def calculate_fd_sip(months, monthly_investment, annual_rate=0.07):
    monthly_rate = annual_rate / 12
    values = []
    for m in range(1, months + 1):
        fv = monthly_investment * (((1 + monthly_rate)**m - 1) / monthly_rate) * (1 + monthly_rate)
        values.append(fv)
    return values

@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(mf_symbol, benchmark_symbol="^NSEI"):
    mf_data = yf.Ticker(mf_symbol).history(period="5y")
    nifty_data = yf.Ticker(benchmark_symbol).history(period="5y")
    return mf_data, nifty_data

# --- MUTUAL FUND DICTIONARY ---
MF_DICT = {
    # --- Flexi Cap / Multi Cap ---
    "Parag Parikh Flexi Cap Fund": "0P0000XVAA.BO",
    "Quant Active Fund": "0P0000XVTR.BO",
    "SBI Flexicap Fund": "0P0000XVWP.BO", 
    
    # --- Small Cap ---
    "Quant Small Cap Fund": "0P0000XVU4.BO",
    "Nippon India Small Cap Fund": "0P0000XVYC.BO",
    "SBI Small Cap Fund": "0P0000XW14.BO",
    "Axis Small Cap Fund": "0P0000XW8F.BO",
    
    # --- Mid Cap ---
    "HDFC Mid-Cap Opportunities Fund": "0P0000XVU0.BO",
    "Kotak Emerging Equity Fund": "0P0000XVWQ.BO",
    "Motilal Oswal Midcap Fund": "0P0000XW6H.BO",
    
    # --- Large Cap & Index ---
    "HDFC Index Fund Nifty 50": "0P0000XVTL.BO",
    "UTI Nifty 50 Index Fund": "0P0000XVXM.BO",
    "Mirae Asset Large Cap Fund": "0P0000XVU6.BO",
    "ICICI Prudential Bluechip Fund": "0P0000XW01.BO",
    
    # --- ELSS (Tax Saving) ---
    "Quant ELSS Tax Saver Fund": "0P0000XVU5.BO",
    "DSP ELSS Tax Saver Fund": "0P0000XVU9.BO",
    
    # --- Custom Fallback ---
    "Custom (Enter Symbol Below)": "CUSTOM"
}

# 1. Setup the Webpage
st.set_page_config(page_title="MF Wealth Engine", page_icon="💰", layout="wide")
st.title("💰 Mutual Fund Wealth Engine")
st.write("CAGR Tracking | SIP Benchmarking (vs Nifty) | Portfolio Rationalizer")
st.divider()

# --- TABS SETUP ---
tab1, tab2 = st.tabs(["📊 Individual Wealth Engine", "🧹 Portfolio Rationalizer (CSV)"])

# ==========================================
# TAB 1: INDIVIDUAL FUND ENGINE
# ==========================================
with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_fund = st.selectbox("Select Mutual Fund", list(MF_DICT.keys()))
    with col2:
        if selected_fund == "Custom (Enter Symbol Below)":
            target_symbol = st.text_input("Enter Yahoo MF Code (e.g., 0P0000XVAA.BO)", value="0P0000XVAA.BO")
        else:
            target_symbol = MF_DICT[selected_fund]
            st.text_input("Yahoo Finance Code (Auto-Filled)", value=target_symbol, disabled=True)
    with col3:
        monthly_budget = st.number_input("Monthly SIP Budget (₹)", value=10000, step=1000)

    send_alert = st.checkbox("📱 Send Telegram Alert on Analysis")

    st.write("<br>", unsafe_allow_html=True)

    if st.button("🔍 Run Wealth Analysis", type="primary"):
        with st.spinner("Crunching Data & Running SIP Simulations..."):
            try:
                mf_hist, nifty_hist = fetch_data(target_symbol)
                
                if len(mf_hist) < 30:
                    st.error("❌ This fund is too new (less than 1 month of data). Mathematical models require at least 30 days of trading history.")
                else:
                    current_nav = mf_hist['Close'].iloc[-1]
                    all_time_high = mf_hist['Close'].max()
                    drawdown_pct = ((current_nav - all_time_high) / all_time_high) * 100
                    
                    daily_returns = mf_hist['Close'].pct_change()
                    annual_volatility = daily_returns.std() * np.sqrt(252) * 100
                    
                    nav_1y_ago = mf_hist['Close'].iloc[-252] if len(mf_hist) >= 252 else None
                    nav_3y_ago = mf_hist['Close'].iloc[-756] if len(mf_hist) >= 756 else None
                    nav_5y_ago = mf_hist['Close'].iloc[0] if len(mf_hist) >= 1260 else None
                    
                    cagr_1y = calculate_cagr(nav_1y_ago, current_nav, 1) if nav_1y_ago else None
                    cagr_3y = calculate_cagr(nav_3y_ago, current_nav, 3) if nav_3y_ago else None
                    
                    if nav_5y_ago:
                        cagr_5y = calculate_cagr(nav_5y_ago, current_nav, 5)
                        cagr_5y_label = "5-Year CAGR"
                    else:
                        days_alive = len(mf_hist)
                        years_alive = days_alive / 252
                        cagr_5y = calculate_cagr(mf_hist['Close'].iloc[0], current_nav, years_alive)
                        cagr_5y_label = "Since Inception CAGR"
                    
                    mf_hist['RSI'] = calculate_rsi(mf_hist['Close'])
                    current_rsi = mf_hist['RSI'].iloc[-1]
                    
                    if len(mf_hist) >= 200:
                        mf_hist['EMA_200'] = mf_hist['Close'].ewm(span=200, adjust=False).mean()
                        current_ema_200 = mf_hist['EMA_200'].iloc[-1]
                    else:
                        current_ema_200 = None
                    
                    st.subheader("🧬 Fund Vital Factors (Mathematical)")
                    
                    v_c1, v_c2, v_c3, v_c4, v_c5 = st.columns(5)
                    v_c1.metric("Current NAV", f"₹{current_nav:.2f}", f"{drawdown_pct:.2f}% from ATH")
                    v_c2.metric("1-Year CAGR", f"{cagr_1y:.2f}%" if cagr_1y is not None else "N/A")
                    v_c3.metric("3-Year CAGR", f"{cagr_3y:.2f}%" if cagr_3y is not None else "N/A")
                    v_c4.metric(cagr_5y_label, f"{cagr_5y:.2f}%")
                    v_c5.metric("Volatility (Risk)", f"{annual_volatility:.2f}%", "Lower is safer" if annual_volatility < 15 else "High Risk", delta_color="inverse")
                    
                    st.divider()

                    st.subheader("💡 Capital Allocation Engine")
                    
                    action_title = ""
                    action_emoji = ""
                    
                    if drawdown_pct <= -15 and current_rsi < 40:
                        action_title = "AGGRESSIVE LUMP SUM"
                        action_emoji = "🎯"
                        st.success(f"{action_emoji} **ACTION: {action_title}.**")
                        st.write(f"The fund is in a deep correction ({drawdown_pct:.2f}%) and mathematically oversold. Deploy your SIP + any extra cash reserves now.")
                    elif current_ema_200 is not None and current_nav < current_ema_200:
                        action_title = "INCREASE SIP"
                        action_emoji = "📈"
                        st.info(f"{action_emoji} **ACTION: {action_title}.**")
                        st.write(f"The NAV is below its 200-Day average. You are buying at a long-term discount. Increase your ₹{monthly_budget} SIP by 20% this month if possible.")
                    elif drawdown_pct > -5 and current_rsi > 70:
                        action_title = "STRICT SIP ONLY"
                        action_emoji = "⚠️"
                        st.warning(f"{action_emoji} **ACTION: {action_title}.**")
                        st.write(f"The fund is overheated near All-Time Highs. Do NOT deploy lump sums here. Stick strictly to your automated ₹{monthly_budget} SIP.")
                    else:
                        action_title = "NORMAL SIP"
                        action_emoji = "🧘"
                        st.info(f"{action_emoji} **ACTION: {action_title}.**")
                        st.write(f"The fund is compounding normally. Continue your standard ₹{monthly_budget} automated SIP.")
                    
                    fund_name_display = selected_fund if selected_fund != "Custom (Enter Symbol Below)" else target_symbol
                    cagr_display = f"{cagr_3y:.1f}%" if cagr_3y is not None else "N/A"
                    
                    telegram_msg = f"{action_emoji} {action_title} | {fund_name_display} | NAV: ₹{current_nav:.2f} | 3Y CAGR: {cagr_display} | Drawdown: {drawdown_pct:.1f}%"
                    
                    st.info(f"**Generated Telegram Push:**\n\n`{telegram_msg}`")
                    
                    if send_alert:
                        bot_token = "8701094564:AAFQER8tQAl2NwGEkKsY1LTV5zUP_7gT4Tg"
                        chat_id = "7927166007"
                        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                        payload = {"chat_id": chat_id, "text": telegram_msg}
                        try:
                            resp = requests.post(url, json=payload)
                            if resp.status_code == 200:
                                st.success("✅ Successfully pushed to Telegram!")
                            else:
                                st.error(f"Failed to send: {resp.text}")
                        except Exception as e:
                            st.error(f"Telegram API Error: {e}")

                    st.divider()

                    st.subheader("🏁 SIP Reality Check: MF vs Nifty 50 vs FD")
                    
                    nifty_hist_matched = nifty_hist.loc[nifty_hist.index >= mf_hist.index[0]]
                    mf_sip = simulate_sip(mf_hist['Close'], monthly_budget)
                    nifty_sip = simulate_sip(nifty_hist_matched['Close'], monthly_budget)
                    
                    total_months = len(mf_sip)
                    fd_values = calculate_fd_sip(total_months, monthly_budget, 0.07)
                    
                    total_invested = mf_sip['Total_Invested'].iloc[-1]
                    mf_final = mf_sip['Portfolio_Value'].iloc[-1]
                    nifty_final = nifty_sip['Portfolio_Value'].iloc[-1] if len(nifty_sip) > 0 else 0
                    fd_final = fd_values[-1] if len(fd_values) > 0 else 0
                    
                    fig = go.Figure()
                    x_axis = mf_sip.index.astype(str)
                    fig.add_trace(go.Scatter(x=x_axis, y=mf_sip['Portfolio_Value'], mode="lines", name=f"Selected Fund (₹{mf_final:,.0f})", line=dict(color="#00BFFF", width=3)))
                    fig.add_trace(go.Scatter(x=x_axis, y=nifty_sip['Portfolio_Value'], mode="lines", name=f"Nifty 50 Index (₹{nifty_final:,.0f})", line=dict(color="#32CD32", width=2)))
                    fig.add_trace(go.Scatter(x=x_axis, y=fd_values, mode="lines", name=f"7% Fixed Deposit (₹{fd_final:,.0f})", line=dict(color="orange", width=2, dash="dash")))
                    fig.add_trace(go.Scatter(x=x_axis, y=mf_sip['Total_Invested'], mode="lines", name=f"Total Invested (₹{total_invested:,.0f})", line=dict(color="gray", width=1, dash="dot")))
                    
                    fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)", font=dict(color="white")), yaxis=dict(tickprefix="₹"))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.write("### 🏆 The Verdict")
                    if mf_final > nifty_final:
                        st.success(f"**Alpha Generator:** This fund successfully beat the Nifty 50 benchmark by **₹{(mf_final - nifty_final):,.0f}** over its lifespan.")
                    else:
                        st.error(f"**Underperformer:** An active investment in this fund lost to a passive Nifty 50 Index fund by **₹{(nifty_final - mf_final):,.0f}**. Re-evaluate if the AMC fees are worth it.")

            except Exception as e:
                st.error(f"An error occurred. Details: {e}")

# ==========================================
# TAB 2: PORTFOLIO RATIONALIZER (CSV UPLOAD)
# ==========================================
with tab2:
    st.header("🧹 Mutual Fund Rationalizer")
    st.write("Upload your Zerodha `holdings.xlsx - Mutual Funds.csv` to instantly detect overlap, eliminate clutter, and optimize your SIPs.")
    
    col_up, col_sip = st.columns(2)
    with col_up:
        uploaded_file = st.file_uploader("Upload Zerodha Coin CSV", type=["csv"])
    with col_sip:
        sip_budget = st.number_input("Target Monthly SIP Budget (₹)", value=20000, step=1000, key="rationalizer_budget")
        
    display_df = pd.DataFrame()
    
    if uploaded_file is not None:
        try:
            # Smart CSV Parser: Scans to ignore the ~22 lines of blank/account details
            raw_df = pd.read_csv(uploaded_file, header=None)
            header_mask = raw_df.apply(lambda row: row.astype(str).str.contains('Instrument|Scheme|Symbol', case=False).any(), axis=1)
            header_row_idx = raw_df[header_mask].index
            
            if len(header_row_idx) > 0:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, skiprows=header_row_idx[0])
            else:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)
            
            # Standardize Columns
            col_mapping = {
                "Instrument": "Fund Name", "Scheme": "Fund Name", "Symbol": "Fund Name",
                "Current value": "Current Value (₹)", "Present Value": "Current Value (₹)",
                "Invested amount": "Invested (₹)", "Invested Value": "Invested (₹)"
            }
            df.rename(columns=col_mapping, inplace=True)
            
            # If standard Investment Value is missing but Avg Price and Qty exist
            if "Invested (₹)" not in df.columns and "Average Price" in df.columns and "Quantity Available" in df.columns:
                df["Invested (₹)"] = df["Average Price"] * df["Quantity Available"]
                
            df = df.dropna(subset=["Fund Name", "Current Value (₹)"])
            display_df = df[["Fund Name", "Invested (₹)", "Current Value (₹)"]].copy()
            
        except Exception as e:
            st.error(f"Error parsing CSV. Please ensure it is the Mutual Funds file. Details: {e}")
            display_df = pd.DataFrame(columns=["Fund Name", "Invested (₹)", "Current Value (₹)"])
    else:
        # Dummy data so the screen isn't empty before upload
        display_df = pd.DataFrame([
            {"Fund Name": "UTI Nifty 50 Index", "Invested (₹)": 169888, "Current Value (₹)": 180262},
            {"Fund Name": "Sundaram Multi Asset", "Invested (₹)": 143222, "Current Value (₹)": 168000},
            {"Fund Name": "JioBlackRock Flexi Cap", "Invested (₹)": 18505, "Current Value (₹)": 18394},
            {"Fund Name": "Navi Nasdaq100 FoF", "Invested (₹)": 142, "Current Value (₹)": 254}
        ])
        
    st.info("💡 **Your Portfolio Data:** Review the extracted data below. You can manually edit numbers before running the algorithm.")
    edited_mf_df = st.data_editor(display_df, num_rows="dynamic", use_container_width=True)
    
    if st.button("Run Rationalization Algorithm", type="primary"):
        with st.spinner("Analyzing portfolio overlap and calculating SIP allocations..."):
            total_current = edited_mf_df['Current Value (₹)'].astype(float).sum()
            results = []
            overlap_flags = {"Index": False, "Flexi": 0, "Large": False}
            clutter_count = 0
            
            # First Pass: Identify categories to check for overlap
            for _, row in edited_mf_df.iterrows():
                name = str(row['Fund Name']).lower()
                if "index" in name or "nifty 50" in name: overlap_flags["Index"] = True
                if "large cap" in name or "bluechip" in name: overlap_flags["Large"] = True
                if "flexi" in name or "multi cap" in name: overlap_flags["Flexi"] += 1
            
            # Second Pass: Generate Verdicts
            for _, row in edited_mf_df.iterrows():
                name = str(row['Fund Name'])
                inv = float(row['Invested (₹)'])
                cur = float(row['Current Value (₹)'])
                
                pl_pct = ((cur - inv) / inv) * 100 if inv > 0 else 0
                pct_of_pf = (cur / total_current) * 100 if total_current > 0 else 0
                
                type_tag = "Equity"
                verdict = "KEEP"
                reason = "Solid allocation."
                name_lower = name.lower()
                
                # Tagging Engine
                if "index" in name_lower or "nifty 50" in name_lower: type_tag = "Index"
                elif "flexi" in name_lower: type_tag = "Flexi Cap"
                elif "large cap" in name_lower or "bluechip" in name_lower: type_tag = "Large Cap"
                elif "multi asset" in name_lower or "balanced" in name_lower or "advantage" in name_lower: type_tag = "Hybrid"
                elif "small cap" in name_lower: type_tag = "Small Cap"
                elif "mid cap" in name_lower: type_tag = "Mid Cap"
                elif "value" in name_lower: type_tag = "Value"
                elif "fof" in name_lower or "nasdaq" in name_lower or "us tech" in name_lower: type_tag = "International"
                
                # Rule 1: Administrative Clutter (Less than ₹10,000)
                if cur < 10000:
                    verdict = "🔴 EXIT (Clutter)"
                    reason = f"Negligible size ({pct_of_pf:.1f}% of PF). Exit to reduce tracking headache."
                    clutter_count += 1
                # Rule 2: Index vs Large Cap Overlap
                elif type_tag == "Large Cap" and overlap_flags["Index"]:
                    verdict = "🔴 EXIT (Overlap)"
                    reason = "Significant overlap with your Nifty 50 Index. Redirect SIP to Index."
                # Rule 3: Redundant Flexi Caps
                elif type_tag == "Flexi Cap" and overlap_flags["Flexi"] > 1 and pct_of_pf < 5:
                    verdict = "🔴 EXIT (Consolidate)"
                    reason = "Small redundant Flexi Cap. Consolidate into your primary one."
                # Rule 4: Core Anchors
                elif type_tag == "Index" and pct_of_pf > 10:
                    verdict = "🟢 KEEP + INCREASE"
                    reason = "Core equity anchor. Lowest cost. Increase SIP here."
                else:
                    verdict = "🟡 HOLD"
                    reason = "Good diversifier. Maintain current allocation."
                    
                results.append({
                    "FUND": name,
                    "TYPE": type_tag,
                    "CURRENT VALUE": f"₹{cur:,.0f}",
                    "P&L %": f"{pl_pct:+.1f}%",
                    "VERDICT": verdict,
                    "REASON": reason
                })
            
            st.divider()
            st.subheader("📋 Executive Summary")
            st.error(f"**The Problem: {len(edited_mf_df)} Funds Analyzed, {clutter_count} Negligible Positions Detected.**\nRationalize to 4–5 funds with clear, non-overlapping mandates to maximize compounding and minimize administrative tracking.")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            
            st.divider()
            st.subheader(f"🎯 Recommended SIP Plan (Budget: ₹{sip_budget:,.0f}/month)")
            st.write("Based on your desired monthly budget, here is the exact mathematical split to achieve a rationalized, 4-fund core portfolio:")
            
            target_pf = [
                {"Fund Role": "Core Nifty 50 Index", "Target Allocation": "40%", "Recommended SIP": f"₹{sip_budget * 0.40:,.0f}"},
                {"Fund Role": "Proven Flexi Cap", "Target Allocation": "30%", "Recommended SIP": f"₹{sip_budget * 0.30:,.0f}"},
                {"Fund Role": "Hybrid / Multi-Asset", "Target Allocation": "20%", "Recommended SIP": f"₹{sip_budget * 0.20:,.0f}"},
                {"Fund Role": "Alpha Generator (Small/Mid Cap)", "Target Allocation": "10%", "Recommended SIP": f"₹{sip_budget * 0.10:,.0f}"}
            ]
            st.table(pd.DataFrame(target_pf))

st.divider()

# --- NFO GAME PLAN & RADAR MODULE ---
with st.expander("📡 Live NFO Radar & Manual Game Plan (April 2026)"):
    st.write("Math engines cannot predict NFOs because they lack historical data. Below is the active research radar for this month, followed by the Fundamental Checklist to evaluate them:")
    
    st.write("### 🗓️ Active NFO Radar (April 2026)")
    nfo_data = [
        {"Fund Name": "Kotak Multi Asset Active FoF", "Category": "Hybrid (Multi-Asset)", "Closes": "Apr 22", "Verdict": "Strong for structured asset allocation. Mechanically diversifies across equity, debt, and gold."},
        {"Fund Name": "Axis Nifty India Defence Index", "Category": "Thematic Equity", "Closes": "Apr 24", "Verdict": "High hype. Very difficult to find underlying stocks here maintaining ROE > 15% and D/E < 1."},
        {"Fund Name": "Groww Arbitrage Fund", "Category": "Hybrid (Arbitrage)", "Closes": "Apr 22", "Verdict": "Low-risk, tax-efficient parking for fresh capital waiting for market dips."},
        {"Fund Name": "SBI CRISIL-IBX Fin. Services", "Category": "Debt Index", "Closes": "Apr 20", "Verdict": "Short-term debt parking (3-12 months)."}
    ]
    st.table(pd.DataFrame(nfo_data))
    
    st.write("### 🛡️ The NFO Scouting Checklist")
    colA, colB, colC = st.columns(3)
    with colA:
        st.info("**1. Examine the AMC**\n\nDoes the parent company have a proven track record? (e.g., Quant, Parag Parikh). A brand new, unknown AMC requires extreme caution.")
    with colB:
        st.info("**2. Verify the Theme**\n\nIs it a broad fund or a highly specific theme (like Defence or EV)? Thematic funds often launch at the absolute peak of a bubble.")
    with colC:
        st.info("**3. Check Expense Ratio**\n\nNFOs start with lower AUM, which can lead to higher early expense ratios. Don't overpay just for novelty.")

# --- MOTIVATIONAL FOOTER ---
st.write("<br><br>", unsafe_allow_html=True)
quotes = [
    "\"Compound interest is the eighth wonder of the world. He who understands it, earns it; he who doesn't, pays it.\" – Albert Einstein",
    "\"The stock market is a device for transferring money from the impatient to the patient.\" – Warren Buffett",
    "\"Time in the market beats timing the market.\"",
    "\"In the short run, the market is a voting machine but in the long run, it is a weighing machine.\" – Benjamin Graham"
]
st.markdown(f"<p style='text-align: center; color: gray;'><i>{random.choice(quotes)}</i></p>", unsafe_allow_html=True)
