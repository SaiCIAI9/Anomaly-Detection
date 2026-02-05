"""
Streamlit Application for Anomaly Analysis
- KPI boxes with key insights
- Filter by granularity level
- Table of anomalies
- Click anomaly to get LLM-powered insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Try to import Azure OpenAI - handle both old and new versions
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    try:
        import openai
        AZURE_OPENAI_AVAILABLE = True
    except ImportError:
        AZURE_OPENAI_AVAILABLE = False
        st.error("OpenAI package not installed. Please install: pip install openai")

# Page config
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----- Password gate (for Streamlit Cloud / internal use) -----
def _get_app_password():
    try:
        return st.secrets.get("APP_PASSWORD", os.environ.get("APP_PASSWORD", "changeme"))
    except Exception:
        return os.environ.get("APP_PASSWORD", "changeme")

if not st.session_state.get("authenticated", False):
    st.title("🔐 Login")
    pwd = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if pwd and pwd == _get_app_password():
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# Initialize Azure OpenAI client
@st.cache_resource
def init_azure_openai():
    """Initialize Azure OpenAI client"""
    try:
        # Get API keys from environment variables or use defaults
        api_key = os.getenv(
            "AZURE_OPENAI_API_KEY",
            "9RRw2jGv9wd77YZA2M0R13UAmltApnU4xCFnRVktSfLrkWPp03XoJQQJ99BFACYeBjFXJ3w3AAABACOG7cCi"
        )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        azure_endpoint = os.getenv(
            "AZURE_OPENAI_ENDPOINT",
            "https://ciathciaiopenai01.openai.azure.com/"
        )
        
        client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
        return client
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI: {str(e)}")
        return None

# Month mapping: T1 = May 2023 (oldest), T25 = May 2025 (newest). Use month names in LLM output.
MONTH_NAMES = [
    'May 2023', 'Jun 2023', 'Jul 2023', 'Aug 2023', 'Sep 2023',
    'Oct 2023', 'Nov 2023', 'Dec 2023', 'Jan 2024', 'Feb 2024',
    'Mar 2024', 'Apr 2024', 'May 2024', 'Jun 2024', 'Jul 2024',
    'Aug 2024', 'Sep 2024', 'Oct 2024', 'Nov 2024', 'Dec 2024',
    'Jan 2025', 'Feb 2025', 'Mar 2025', 'Apr 2025', 'May 2025'
]
# T1..T25 map to MONTH_NAMES[0]..[24]. E.g. T24 = Apr 2025, T25 = May 2025.
T_TO_MONTH = {i + 1: MONTH_NAMES[i] for i in range(25)}
# Give LLM a short mapping so it uses month names (Apr 2025, May 2025) not "T24"/"T25"
TIME_PERIODS_FOR_LLM = "T1=May 2023, T9=Jan 2024, T12=Apr 2024, T18=Oct 2024, T20=Dec 2024, T24=Apr 2025, T25=May 2025"

# Default LLM prompt template (used when no stored prompt). Placeholders: {TIME_PERIODS_FOR_LLM}, {full_row_str}
DEFAULT_LLM_PROMPT = """You are a pharmaceutical market access analyst. Below is one full row for a payer flagged as an anomaly.

TIME PERIODS (use month names in your response, not T24/T25): {TIME_PERIODS_FOR_LLM}. So say "Apr 2025" or "May 2025", not "T24" or "T25". For other T values, infer the month (e.g. T20 = Dec 2024).

DATA (one row, key: value):
{full_row_str}

Rules: Be concise. Lead with one sentence + one number + one action. Do not repeat the same metric in multiple sections. Cite numbers for every claim.

Provide:
1. TITLE: Short title (max 10 words).
2. KEY INSIGHT: One sentence: what is wrong, one headline number, one priority action. Then optionally 1–2 more short sentences with 1–2 numbers. No long paragraphs.
3. SUMMARY: 2–3 short takeaways (volume/trend, flags, what to do). One line each. No repetition of KEY INSIGHT.
4. KEY POINTS: Short bullets only (one line each). Include numbers. Do not repeat the same metric in both Volume & Trends and Root Causes.
   - Volume & Trends: 2–3 bullets (TRx/NBRx levels or growth, volatility, trend). Use month names (e.g. May 2025, Apr 2025).
   - Root causes: 2–3 bullets; each = one cause + one supporting number (e.g. "NBRx/TRx in Apr 2025 = 0.07, below 12M mean → formulary risk").
   - Actions: 2–3 short bullets (e.g. "Review formulary for May 2025"; "Check HCP engagement").

Format exactly:
TITLE: [title]
KEY INSIGHT: [one sentence + number + action; then optional 1–2 sentences]
SUMMARY: [2–3 short lines]
KEY POINTS:
- [short bullets only]
"""
DEFAULT_SYSTEM_MESSAGE = "You are a senior pharmaceutical market access analyst. Be concise: lead with one sentence, one number, one action. Use month names (e.g. Apr 2025, May 2025, Dec 2024)—never refer to T24, T25, or T20 in the final text. Cite specific metrics; list 2–3 root causes with one supporting number each; do not repeat the same metric in multiple sections."

# ----- Prompt storage (Supabase for Streamlit Cloud; optional) -----
def _supabase_client():
    try:
        if hasattr(st, "secrets") and st.secrets.get("SUPABASE_URL") and st.secrets.get("SUPABASE_KEY"):
            from supabase import create_client
            return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
    except Exception:
        pass
    return None

def get_prompt():
    """Load prompt: session override (from Save) > Supabase > default. Session keeps edited prompt across reruns."""
    if st.session_state.get("llm_prompt_override"):
        return st.session_state["llm_prompt_override"]
    client = _supabase_client()
    if client:
        try:
            r = client.table("app_config").select("value").eq("key", "llm_prompt").limit(1).execute()
            if r.data and len(r.data) > 0 and r.data[0].get("value"):
                return r.data[0]["value"]
        except Exception:
            pass
    return DEFAULT_LLM_PROMPT

def save_prompt(text):
    """Save prompt to Supabase. Returns True if saved, False otherwise."""
    client = _supabase_client()
    if client:
        try:
            client.table("app_config").upsert({"key": "llm_prompt", "value": text}, on_conflict="key").execute()
            return True
        except Exception:
            return False
    return False

def prompt_storage_available():
    return _supabase_client() is not None

# Load data (Anomalies_List.csv = GAN T24 results joined to full ML dataset)
ANOMALY_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anomalies_List.csv')
if not os.path.exists(ANOMALY_CSV):
    ANOMALY_CSV = 'Anomalies_List.csv'

@st.cache_data
def load_data():
    """Load anomaly dataset (Anomalies_List.csv: payer-level with all features + GAN columns)"""
    df = pd.read_csv(ANOMALY_CSV, low_memory=False)
    # Derive columns for app compatibility: entity, granularity, anomaly flag, priority
    if 'PRCSN_PAYER_ENT_NM' in df.columns and 'GRANULARITY' not in df.columns:
        df['GRANULARITY'] = df['PRCSN_PAYER_ENT_NM']
    if 'Priority' in df.columns and 'Anomaly_Priority' not in df.columns:
        df['Anomaly_Priority'] = df['Priority'].astype(str).str.strip()
    # Normalize priority so sidebar filter matches (Medium -> Med, etc.)
    if 'Anomaly_Priority' in df.columns:
        df['Anomaly_Priority'] = df['Anomaly_Priority'].replace({'Medium': 'Med', 'High': 'High', 'Low': 'Low', 'Normal': 'Normal'})
    # Show all non-Normal priorities in the anomalies table (High, Med, Low), not only Threshold_Flag==1
    priority_col = df.get('Anomaly_Priority', df.get('Priority', pd.Series(['Normal'] * len(df))))
    norm_priority = priority_col.fillna('Normal').astype(str).str.strip().replace({'Medium': 'Med'})
    df['Is_Anomaly'] = (norm_priority != 'Normal')
    df['GRANULARITY_LEVEL'] = 'Payer'
    return df

# Initialize
if AZURE_OPENAI_AVAILABLE:
    client = init_azure_openai()
else:
    client = None
df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

granularity_levels = ['All'] + sorted(df['GRANULARITY_LEVEL'].dropna().unique().tolist())
selected_level = st.sidebar.selectbox("Granularity Level", granularity_levels)

priority_levels = ['All', 'High', 'Med', 'Low', 'Normal']
selected_priority = st.sidebar.selectbox("Priority", priority_levels)

# Edit prompt (stored in Supabase on Streamlit Cloud; team can improve and save)
with st.sidebar.expander("✏️ Edit prompt", expanded=False):
    if not prompt_storage_available():
        st.caption("Add SUPABASE_URL and SUPABASE_KEY to secrets to persist prompt.")
    current_prompt = get_prompt()
    edited = st.text_area("LLM prompt template", value=current_prompt, height=200, key="prompt_editor",
                          help="Use {TIME_PERIODS_FOR_LLM} and {full_row_str} as placeholders.")
    if st.button("Save prompt", key="save_prompt_btn"):
        st.session_state["llm_prompt_override"] = edited  # use for this session across reruns
        if save_prompt(edited):
            st.success("Prompt saved. It will be used for the next analysis.")
        else:
            st.info("Saved for this session only (Supabase not configured). Your prompt will be used until you refresh.")

# Filter data
filtered_df = df.copy()
if selected_level != 'All':
    filtered_df = filtered_df[filtered_df['GRANULARITY_LEVEL'] == selected_level]
if selected_priority != 'All':
    filtered_df = filtered_df[filtered_df['Anomaly_Priority'] == selected_priority]

# Main page
st.title("🔍 Anomaly Detection Dashboard")
st.markdown("---")

# KPI Boxes
st.header("📊 Key Insights")

col1, col2, col3, col4 = st.columns(4)

total_anomalies = filtered_df['Is_Anomaly'].sum()
total_entities = len(filtered_df)
anomaly_rate = (total_anomalies / total_entities * 100) if total_entities > 0 else 0

high_priority = (filtered_df['Anomaly_Priority'] == 'High').sum() if 'Anomaly_Priority' in filtered_df.columns else 0
spike_anomalies = filtered_df['Flag_Spike_Above_20'].sum() if 'Flag_Spike_Above_20' in filtered_df.columns else 0
relationship_anomalies = filtered_df['Flag_Relationship_Violation'].sum() if 'Flag_Relationship_Violation' in filtered_df.columns else 0

with col1:
    st.metric(
        label="Total Anomalies",
        value=f"{total_anomalies:,}",
        delta=f"{anomaly_rate:.2f}% of entities"
    )

with col2:
    st.metric(
        label="High Priority",
        value=f"{high_priority:,}",
        delta="Critical anomalies"
    )

with col3:
    st.metric(
        label="Spike Anomalies",
        value=f"{spike_anomalies:,}",
        delta=">20% T24→T25 change"
    )

with col4:
    st.metric(
        label="Relationship Violations",
        value=f"{relationship_anomalies:,}",
        delta="Metric pattern violations"
    )

# Breakdown by granularity level
st.markdown("---")
st.header("📈 Anomaly Distribution")

col1, col2 = st.columns(2)

with col1:
    # By granularity level
    if 'GRANULARITY_LEVEL' in filtered_df.columns:
        level_counts = filtered_df.groupby('GRANULARITY_LEVEL').agg({
            'Is_Anomaly': ['sum', 'count']
        }).reset_index()
        level_counts.columns = ['Granularity_Level', 'Anomalies', 'Total']
        level_counts['Rate'] = (level_counts['Anomalies'] / level_counts['Total'] * 100).round(2)
        
        fig_levels = go.Figure(data=[
            go.Bar(
                x=level_counts['Granularity_Level'],
                y=level_counts['Anomalies'],
                text=level_counts['Rate'].apply(lambda x: f"{x}%"),
                textposition='auto',
                marker_color='#FF6B6B'
            )
        ])
        fig_levels.update_layout(
            title="Anomalies by Granularity Level",
            xaxis_title="Granularity Level",
            yaxis_title="Number of Anomalies",
            height=300
        )
        st.plotly_chart(fig_levels, use_container_width=True)

with col2:
    # By priority
    if 'Anomaly_Priority' in filtered_df.columns:
        priority_counts = filtered_df[filtered_df['Is_Anomaly'] == True]['Anomaly_Priority'].value_counts()
        
        fig_priority = go.Figure(data=[
            go.Bar(
                x=priority_counts.index,
                y=priority_counts.values,
                marker_color=['#FF4444', '#FFAA00', '#FFCC99'],
                text=priority_counts.values,
                textposition='auto'
            )
        ])
        fig_priority.update_layout(
            title="Anomalies by Priority",
            xaxis_title="Priority",
            yaxis_title="Count",
            height=300
        )
        st.plotly_chart(fig_priority, use_container_width=True)

# Anomalies Table
st.markdown("---")
st.header("📋 Anomalies Table")

# Prepare table data
anomalies_df = filtered_df[filtered_df['Is_Anomaly'] == True].copy()
total_entities_filtered = len(filtered_df)
total_anomalies_filtered = len(anomalies_df)

# Show count so user knows why only N visible (data vs filter)
st.caption(f"Showing **{total_anomalies_filtered}** anomaly(ies) out of **{total_entities_filtered}** entities. Set **Priority** and **Granularity Level** to *All* in the sidebar to see every flagged anomaly.")

if len(anomalies_df) > 0:
    # Select columns for display (Anomalies_List schema: entity, priority, GAN scores, errors, explanation)
    display_cols = [
        'PRCSN_PAYER_ENT_NM', 'GRANULARITY_LEVEL', 'Anomaly_Priority',
        'Discriminator_Score', 'TRx_T24_actual', 'TRx_T24_pred', 'TRx_T24_PctError',
        'Threshold_Score', 'Explanation'
    ]
    display_cols = [c for c in display_cols if c in anomalies_df.columns]
    if not display_cols:
        display_cols = [c for c in ['PRCSN_PAYER_ENT_NM', 'GRANULARITY', 'Anomaly_Priority', 'Explanation'] if c in anomalies_df.columns]
    
    table_df = anomalies_df[display_cols].copy()
    
    if 'TRx_T24_PctError' in table_df.columns:
        table_df['TRx_T24_PctError'] = table_df['TRx_T24_PctError'].apply(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "")
    
    # Add row numbers
    table_df.insert(0, 'Row', range(1, len(table_df) + 1))
    
    # Display table
    st.dataframe(
        table_df,
        use_container_width=True,
        height=400
    )
    
    # Row selection dropdown
    st.markdown("### Select Anomaly for AI Analysis")
    
    # Reset index to ensure positional indexing works
    anomalies_df_reset = anomalies_df.reset_index(drop=True)
    
    # Create selection options (Anomalies_List: PRCSN_PAYER_ENT_NM, Priority, Threshold_Score)
    anomaly_options = []
    for pos_idx in range(len(anomalies_df_reset)):
        row = anomalies_df_reset.iloc[pos_idx]
        entity = row.get('PRCSN_PAYER_ENT_NM', row.get('GRANULARITY', 'Unknown'))
        level = row.get('GRANULARITY_LEVEL', 'Payer')
        priority = row.get('Anomaly_Priority', row.get('Priority', 'Unknown'))
        score = row.get('Threshold_Score', row.get('Anomaly_Score', 0))
        option_label = f"{entity} ({level}) - Priority: {priority}, Score: {score}"
        anomaly_options.append((pos_idx, option_label))
    
    # Create selectbox
    selected_option = st.selectbox(
        "Choose an anomaly to analyze:",
        options=range(len(anomaly_options)),
        format_func=lambda x: anomaly_options[x][1],
        key="anomaly_selector"
    )
    
    # Get selected anomaly using positional index
    if selected_option is not None and selected_option < len(anomalies_df_reset):
        selected_anomaly = anomalies_df_reset.iloc[selected_option]
        
        # Store in session state
        st.session_state['selected_anomaly'] = selected_anomaly.to_dict()
        st.session_state['selected_row_idx'] = selected_option
        
        # Button to view details
        if st.button("🔍 Analyze This Anomaly with AI", type="primary", use_container_width=True):
            st.session_state['show_details'] = True
            st.rerun()

# Anomaly Details Page
if st.session_state.get('show_details', False) and 'selected_anomaly' in st.session_state:
    st.markdown("---")
    st.header("🤖 AI-Powered Anomaly Analysis")
    
    selected_anomaly = st.session_state['selected_anomaly']
    
    entity_name = selected_anomaly.get('PRCSN_PAYER_ENT_NM', selected_anomaly.get('GRANULARITY', 'Unknown'))
    granularity_level = selected_anomaly.get('GRANULARITY_LEVEL', 'Payer')
    
    # Time series for charts (Anomalies_List: TRx_T1..T25, NBRx_T1..T25, NRx_T1..T25)
    trx_ts = []
    nbrx_ts = []
    nrx_ts = []
    for i in range(1, 26):
        try:
            v = selected_anomaly.get(f'TRx_T{i}')
            trx_ts.append(float(v) if pd.notna(v) else 0.0)
        except (ValueError, TypeError):
            trx_ts.append(0.0)
        try:
            v = selected_anomaly.get(f'NBRx_T{i}')
            nbrx_ts.append(float(v) if pd.notna(v) else 0.0)
        except (ValueError, TypeError):
            nbrx_ts.append(0.0)
        try:
            v = selected_anomaly.get(f'NRx_T{i}')
            nrx_ts.append(float(v) if pd.notna(v) else 0.0)
        except (ValueError, TypeError):
            nrx_ts.append(0.0)
    
    anomaly_data = {
        'Entity': entity_name,
        'Granularity_Level': granularity_level,
        'Anomaly_Score': selected_anomaly.get('Threshold_Score', selected_anomaly.get('Anomaly_Score', 0)),
        'Anomaly_Priority': selected_anomaly.get('Anomaly_Priority', selected_anomaly.get('Priority', 'Unknown')),
        'T24_to_T25_Change': selected_anomaly.get('TRx_T24_PctError', selected_anomaly.get('T24_to_T25_PctChange', 0)),
        'TRx_T24': selected_anomaly.get('TRx_T24_actual', selected_anomaly.get('TRx_T24', 0)),
        'TRx_T25': selected_anomaly.get('TRx_T25', selected_anomaly.get('T25_TRx', 0)),
        'Volatility_Category': selected_anomaly.get('TRx_Volatility_Category', 'Unknown'),
        'Trend_Direction': selected_anomaly.get('TRx_Trend_Direction', 'Unknown'),
        'Anomaly_Explanation': selected_anomaly.get('Explanation', selected_anomaly.get('Anomaly_Explanation', 'No explanation')),
    }
    
    # Build full row as string for LLM (all features in one row)
    def _row_str(d, max_len=100):
        lines = []
        for k, v in d.items():
            if pd.isna(v) or v == '':
                continue
            try:
                s = f"{k}: {v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}: {v}"
            except Exception:
                s = f"{k}: {v}"
            if len(s) > max_len:
                s = s[:max_len] + "..."
            lines.append(s)
        return "\n".join(lines)
    
    full_row_str = _row_str(selected_anomaly)
    
    # Call LLM for analysis
    if client is None:
        st.warning("⚠️ Azure OpenAI client not available. Using default analysis.")
        title = f"Anomaly: {entity_name}"
        key_insight = anomaly_data['Anomaly_Explanation']
        summary = f"This entity shows an anomaly with score {anomaly_data['Anomaly_Score']}. {anomaly_data['Anomaly_Explanation']}"
        key_points = ""
    else:
        with st.spinner("🤖 AI is analyzing the anomaly..."):
            prompt_template = get_prompt()
            try:
                prompt = prompt_template.format(TIME_PERIODS_FOR_LLM=TIME_PERIODS_FOR_LLM, full_row_str=full_row_str)
            except KeyError:
                prompt = prompt_template.replace("{TIME_PERIODS_FOR_LLM}", TIME_PERIODS_FOR_LLM).replace("{full_row_str}", full_row_str)
            try:
                response = client.chat.completions.create(
                    model="ciathena-gpt-4o",
                    messages=[
                        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000
                )
                
                llm_response = (response.choices[0].message.content or "").strip()
                
                # Parse LLM response (support both old format and custom prompt formats)
                title = "Anomaly Analysis"
                key_insight = "Analysis in progress..."
                summary = "Processing..."
                key_points = ""
                
                if "TITLE:" in llm_response and "KEY INSIGHT:" in llm_response:
                    if "TITLE:" in llm_response:
                        title = llm_response.split("TITLE:")[1].split("KEY INSIGHT:")[0].strip()
                    if "KEY INSIGHT:" in llm_response:
                        key_insight = llm_response.split("KEY INSIGHT:")[1].split("SUMMARY:")[0].strip()
                    if "SUMMARY:" in llm_response:
                        if "KEY POINTS:" in llm_response:
                            summary = llm_response.split("SUMMARY:")[1].split("KEY POINTS:")[0].strip()
                            key_points = llm_response.split("KEY POINTS:")[1].strip()
                        else:
                            summary = llm_response.split("SUMMARY:")[1].strip()
                else:
                    # Custom prompt format (e.g. Opening paragraph, Why this matters now, Bottom line): show full response
                    title = f"Anomaly: {entity_name}"
                    key_insight = llm_response if llm_response else anomaly_data['Anomaly_Explanation']
                    summary = ""
                    key_points = ""
                
                # Whatever the prompt format, always show the model output (no stuck placeholders)
                if llm_response and (key_insight == "Analysis in progress..." or not key_insight.strip()):
                    key_insight = llm_response
                    summary = ""
                if llm_response and summary == "Processing...":
                    summary = ""
                
            except Exception as e:
                st.error(f"Error calling LLM: {str(e)}")
                title = f"Anomaly: {entity_name}"
                key_insight = anomaly_data['Anomaly_Explanation']
                summary = f"This entity shows an anomaly with score {anomaly_data['Anomaly_Score']}. {anomaly_data['Anomaly_Explanation']}"
                key_points = ""
    
    # Display LLM Analysis (remove markdown formatting from title)
    clean_title = title.replace('**', '').replace('*', '').strip()
    st.subheader(f"🤖 AI-Powered Anomaly Analysis: {clean_title}")
    
    st.markdown("### 💡 Key Insight")
    st.info(key_insight)
    
    if summary:
        st.markdown("### 📝 Comprehensive Strategic Analysis")
        st.markdown(summary)
    
    # Display KEY POINTS if available
    if key_points:
        st.markdown("### 🔑 Key Points")
        st.markdown(key_points)
    
    # Trend Lines
    st.markdown("### 📈 Trend Analysis")
    
    # Create subplots for each metric
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('TRx Trend', 'NBRx Trend', 'NRx Trend'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Use actual month names in chronological order (oldest to newest, left to right)
    # T1 = May 2023 (oldest), T25 = May 2025 (newest/predicted)
    # MONTH_NAMES is already in chronological order (T1 to T25)
    time_periods = MONTH_NAMES  # May 2023 (left) to May 2025 (right)
    
    # Data is already in T1 to T25 order (trx_ts[0] = T1 = May 2023, trx_ts[24] = T25 = May 2025)
    # No need to reverse - already in chronological order
    trx_ts_chrono = trx_ts  # T1 to T25: [May 2023, ..., May 2025]
    nbrx_ts_chrono = nbrx_ts  # T1 to T25
    nrx_ts_chrono = nrx_ts  # T1 to T25
    
    # TRx trend
    fig.add_trace(
        go.Scatter(
            x=time_periods,
            y=trx_ts_chrono,
            mode='lines+markers',
            name='TRx',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Highlight T25 (May 2025) - the predicted/latest month where anomaly occurs
    # In chronological order, T25 is at the end (right side)
    fig.add_trace(
        go.Scatter(
            x=['May 2025'],  # T25 - the predicted month
            y=[trx_ts_chrono[24]],  # Last value = T25
            mode='markers',
            name='Anomaly Period (May 2025)',
            marker=dict(color='red', size=12, symbol='diamond'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # NBRx trend
    fig.add_trace(
        go.Scatter(
            x=time_periods,
            y=nbrx_ts_chrono,
            mode='lines+markers',
            name='NBRx',
            line=dict(color='#2ca02c', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    # NRx trend
    fig.add_trace(
        go.Scatter(
            x=time_periods,
            y=nrx_ts_chrono,
            mode='lines+markers',
            name='NRx',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=4)
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text="Time Series Trends (May 2023 to May 2025)",
        showlegend=True
    )
    
    # Update x-axes to show all month labels
    fig.update_xaxes(
        title_text="Month",
        row=1, col=1,
        tickangle=-45,
        tickmode='linear',
        tickvals=time_periods,
        ticktext=time_periods
    )
    fig.update_xaxes(
        title_text="Month",
        row=2, col=1,
        tickangle=-45,
        tickmode='linear',
        tickvals=time_periods,
        ticktext=time_periods
    )
    fig.update_xaxes(
        title_text="Month",
        row=3, col=1,
        tickangle=-45,
        tickmode='linear',
        tickvals=time_periods,
        ticktext=time_periods
    )
    fig.update_yaxes(title_text="TRx Volume", row=1, col=1)
    fig.update_yaxes(title_text="NBRx Volume", row=2, col=1)
    fig.update_yaxes(title_text="NRx Volume", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly Details Table
    st.markdown("### 📊 Anomaly Details")
    
    t24_t25_val = anomaly_data['T24_to_T25_Change']
    try:
        t24_t25_str = f"{float(t24_t25_val):.2f}%" if pd.notna(t24_t25_val) else "N/A"
    except (ValueError, TypeError):
        t24_t25_str = str(t24_t25_val)
    details_data = {
        'Metric': ['Entity', 'Granularity Level', 'Threshold Score', 'Priority',
                   'TRx T24 % Error', 'TRx T24', 'TRx T25', 'Volatility', 'Trend'],
        'Value': [
            entity_name,
            granularity_level,
            f"{anomaly_data['Anomaly_Score']}/4",
            anomaly_data['Anomaly_Priority'],
            t24_t25_str,
            f"{float(anomaly_data['TRx_T24']):,.0f}" if pd.notna(anomaly_data['TRx_T24']) else "N/A",
            f"{float(anomaly_data['TRx_T25']):,.0f}" if pd.notna(anomaly_data['TRx_T25']) else "N/A",
            anomaly_data['Volatility_Category'],
            anomaly_data['Trend_Direction']
        ]
    }
    
    details_df = pd.DataFrame(details_data)
    st.table(details_df)
    
    # Back button
    if st.button("← Back to Anomalies Table"):
        st.session_state['show_details'] = False
        if 'selected_anomaly' in st.session_state:
            del st.session_state['selected_anomaly']
        st.rerun()

# Footer
st.markdown("---")
st.markdown("**Anomaly Detection Dashboard** | Powered by GAN + LLM Analysis")

