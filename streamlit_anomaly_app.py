"""
Streamlit Application for Anomaly Analysis
- KPI boxes with key insights
- Filter by granularity level
- Table of anomalies
- Click anomaly to get LLM-powered insights
"""

import re
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
# Give LLM a short mapping so it uses month names
TIME_PERIODS_FOR_LLM = "T1=May 2023, T9=Jan 2024, T12=Apr 2024, T17=Sep 2024, T18=Oct 2024, T19=Nov 2024, T20=Dec 2024, T24=Apr 2025, T25=May 2025"

# Prompt with dataset context, time context, anomaly context
DEFAULT_LLM_PROMPT = """You are a senior pharmaceutical market access strategist. Below is the context and full row for a payer-PBM flagged as an anomaly.

--- CONTEXT ---
{dataset_context}

{time_context}

{anomaly_context}

--- FULL ROW DATA (analyze the entire row for patterns) ---
{full_row_str}

--- YOUR TASK ---
Analyze the entire row for patterns. Drill down using relationships (NBRx/TRx, formulary, HCPs, states, MoM), all features, and find reasons why TRx dropped. Use Geoff-style language: strategic, not number dumps. Lead with the insight that would change strategy.

Provide:
1. TITLE: Short strategic title (max 10 words). Not "TRx Drop" but something like "Conversion Gap in IA/NE" or "HCP Disengagement in Top State".
2. ONE-LINER: The ground-breaking insight in one sentence. Lead with the strategic implication.
3. KEY INSIGHT: What would surprise an analyst? One sentence. One supporting fact. One concrete action.
4. SUMMARY: 2–3 strategic takeaways. What broke? What to do?
5. KEY DRIVERS: List exactly 5 key drivers that explain the TRx drop. Analyze the entire row (TRx, NBRx, NRx, HCPs, states, formulary, MoM, shares, etc.) and identify the top 5 factors. CRITICAL: When citing metrics, use the month name so readers understand (e.g., "HCPs in Oct 2024" or "TRx (Oct 2024)" instead of "HCPs_T18"). Explain what T18 means: T18 = October 2024. Same for T17 = September 2024, T16 = August 2024.
6. KEY RELATIONSHIP VIOLATIONS: List 2–3 relationship violations. These are metric pairs or ratios that broke expected patterns (e.g., NBRx/TRx ratio outside historical range, HCP count vs TRx mismatch, state share vs expected, formulary vs volume disconnect). For each, cite the metric and explain the violation. Use month names (Oct 2024, Sep 2024), not T18/T17.
7. KEY POINTS: Short bullets. Focus on patterns and causes.
   - What the data shows (pattern).
   - Root cause (formulary? HCP? conversion? state mix?).
   - Action (specific: which state, formulary, or check).

Format exactly:
TITLE: [title]
ONE-LINER: [strategic insight]
KEY INSIGHT: [what would surprise an analyst]
SUMMARY: [2–3 strategic lines]
KEY DRIVERS:
1. [driver 1 - use month names, e.g., HCPs in Oct 2024: 10]
2. [driver 2]
3. [driver 3]
4. [driver 4]
5. [driver 5]
KEY RELATIONSHIP VIOLATIONS:
1. [violation 1 - metric pair/ratio that broke, e.g., NBRx/TRx ratio in Oct 2024 outside expected range]
2. [violation 2]
KEY POINTS:
- [short bullets]
"""
DEFAULT_SYSTEM_MESSAGE = "You are a senior pharmaceutical market access strategist. Your audience (Geoff, Binder) wants strategic insights, not number dumps. Use month names (Oct 2024, Sep 2024) in all output—never T17 or T18. When citing metrics, say 'HCPs in Oct 2024' or 'TRx (Oct 2024)' so readers understand; T18 = October 2024, T17 = September 2024. Include metric names so the reader knows which field you mean."

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

# Data sources: payer-level T24 vs payer-PBM T18 (prefer WITH_SOP_NVS when available)
ANOMALY_CSV_PAYER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anomalies_List.csv')
ANOMALY_CSV_PAYER_PBM = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anomalies_List_Payer_PBM_T18.csv')
ANOMALY_CSV_PAYER_PBM_SOP_NVS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anomalies_List_Payer_PBM_T18_WITH_SOP_NVS.csv')

@st.cache_data
def load_data(csv_path=None):
    """Load anomaly dataset (payer-level T24 or payer-PBM T18 with all features + GAN columns)"""
    path = csv_path or (ANOMALY_CSV_PAYER_PBM if os.path.exists(ANOMALY_CSV_PAYER_PBM) else ANOMALY_CSV_PAYER)
    if not os.path.exists(path):
        path = ANOMALY_CSV_PAYER
    df = pd.read_csv(path, low_memory=False)
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
    if 'GRANULARITY_LEVEL' not in df.columns:
        df['GRANULARITY_LEVEL'] = 'Payer-PBM' if 'PRCSN_PBM_VENDOR' in df.columns else 'Payer'
    return df

# Initialize
if AZURE_OPENAI_AVAILABLE:
    client = init_azure_openai()
else:
    client = None

# Data source selector
st.sidebar.header("Data Source")
data_options = []
if os.path.exists(ANOMALY_CSV_PAYER_PBM_SOP_NVS):
    data_options.append(("Payer-PBM T18 + SOP/NVS (Oct 2024)", ANOMALY_CSV_PAYER_PBM_SOP_NVS))
elif os.path.exists(ANOMALY_CSV_PAYER_PBM):
    data_options.append(("Payer-PBM T18 (Oct 2024)", ANOMALY_CSV_PAYER_PBM))
if os.path.exists(ANOMALY_CSV_PAYER):
    data_options.append(("Payer T24 (Apr 2025)", ANOMALY_CSV_PAYER))
selected_data_label = st.sidebar.selectbox(
    "Dataset",
    options=[o[0] for o in data_options],
    index=0,
    key="data_source"
)
selected_csv = next((p for l, p in data_options if l == selected_data_label), ANOMALY_CSV_PAYER_PBM_SOP_NVS if os.path.exists(ANOMALY_CSV_PAYER_PBM_SOP_NVS) else (ANOMALY_CSV_PAYER_PBM if os.path.exists(ANOMALY_CSV_PAYER_PBM) else ANOMALY_CSV_PAYER))
df = load_data(selected_csv)
is_payer_pbm = 'PRCSN_PBM_VENDOR' in df.columns and selected_csv in (ANOMALY_CSV_PAYER_PBM, ANOMALY_CSV_PAYER_PBM_SOP_NVS)
target_period = 'T18' if is_payer_pbm else 'T24'

# Sidebar filters
st.sidebar.header("Filters")

granularity_levels = ['All'] + sorted(df['GRANULARITY_LEVEL'].dropna().unique().tolist())
selected_level = st.sidebar.selectbox("Granularity Level", granularity_levels)

priority_levels = ['All', 'High', 'Med', 'Low', 'Normal']
selected_priority = st.sidebar.selectbox("Priority", priority_levels)

# SOP/NVS filters (when columns exist)
sop_filter = 'All'
nvs_filter = 'All'
if 'SOP_Pass_Count' in df.columns:
    sop_filter = st.sidebar.selectbox("SOP Confirmed (3+ of 4)", ['All', 'Yes (3+ pass)', 'No'], key="sop_filter")
if 'Novartis_Any_Flag' in df.columns:
    nvs_filter = st.sidebar.selectbox("Novartis Flag", ['All', 'Yes', 'No'], key="nvs_filter")

# Edit prompt (stored in Supabase on Streamlit Cloud; team can improve and save)
with st.sidebar.expander("✏️ Edit prompt", expanded=False):
    if not prompt_storage_available():
        st.caption("Add SUPABASE_URL and SUPABASE_KEY to secrets to persist prompt.")
    current_prompt = get_prompt()
    edited = st.text_area("LLM prompt template", value=current_prompt, height=220, key="prompt_editor",
                          help="Placeholders: {dataset_context}, {time_context}, {anomaly_context}, {full_row_str}")
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
if sop_filter == 'Yes (3+ pass)' and 'GAN_Anomaly_Confirmed_3of4' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['GAN_Anomaly_Confirmed_3of4'] == 1]
elif sop_filter == 'No' and 'GAN_Anomaly_Confirmed_3of4' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['GAN_Anomaly_Confirmed_3of4'] != 1]
if nvs_filter == 'Yes' and 'Novartis_Any_Flag' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Novartis_Any_Flag'] == 1]
elif nvs_filter == 'No' and 'Novartis_Any_Flag' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['Novartis_Any_Flag'] != 1]

# Payer-PBM with SOP/NVS: filter to Anomaly_Flag=1 AND Novartis_Any_Flag=1 (all records, T1-T18 focus)
if is_payer_pbm and 'Anomaly_Flag' in filtered_df.columns and 'Novartis_Any_Flag' in filtered_df.columns:
    filtered_df = filtered_df[(filtered_df['Anomaly_Flag'] == 1) & (filtered_df['Novartis_Any_Flag'] == 1)]

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
        delta="Prediction error above threshold"
    )

with col4:
    st.metric(
        label="Relationship Violations",
        value=f"{relationship_anomalies:,}",
        delta="Metric pattern violations"
    )

# SOP/NVS KPIs when available
if 'SOP_Pass_Count' in filtered_df.columns or 'Novartis_Any_Flag' in filtered_df.columns:
    st.markdown("#### SOP & Novartis Validation")
    c1, c2, c3 = st.columns(3)
    with c1:
        sop3 = (filtered_df['GAN_Anomaly_Confirmed_3of4'] == 1).sum() if 'GAN_Anomaly_Confirmed_3of4' in filtered_df.columns else 0
        st.metric("SOP 3+ of 4 Pass", f"{sop3:,}", "Business-valid anomalies")
    with c2:
        nvs = filtered_df['Novartis_Any_Flag'].sum() if 'Novartis_Any_Flag' in filtered_df.columns else 0
        st.metric("Novartis Flag", f"{nvs:,}", "Trend break / IF / LOF")
    with c3:
        both = ((filtered_df['GAN_Anomaly_Confirmed_3of4'] == 1) & (filtered_df['Novartis_Any_Flag'] == 1)).sum() if all(c in filtered_df.columns for c in ['GAN_Anomaly_Confirmed_3of4', 'Novartis_Any_Flag']) else 0
        st.metric("SOP 3+ and NVS", f"{both:,}", "Strengthened alerts")

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

# Prepare table data. When filtered to Anomaly_Flag=1 & Novartis_Any_Flag=1, all rows are anomalies
if is_payer_pbm and 'Anomaly_Flag' in filtered_df.columns and 'Novartis_Any_Flag' in filtered_df.columns:
    anomalies_df = filtered_df.copy()
else:
    anomalies_df = filtered_df[filtered_df['Is_Anomaly'] == True].copy()
total_entities_filtered = len(filtered_df)
total_anomalies_filtered = len(anomalies_df)

# Show count so user knows why only N visible (data vs filter)
filter_note = " (Payer-PBM: filtered to Anomaly_Flag=1 and Novartis_Any_Flag=1)" if (is_payer_pbm and 'Novartis_Any_Flag' in df.columns) else ""
st.caption(f"Showing **{total_anomalies_filtered}** anomaly(ies) out of **{total_entities_filtered}** entities.{filter_note} Set **Priority** and **Granularity Level** to *All* in the sidebar to see every flagged anomaly.")

if len(anomalies_df) > 0:
    # Select columns for display (T18 for payer-PBM, T24 for payer)
    trx_actual = f'TRx_{target_period}_actual'
    trx_pred = f'TRx_{target_period}_pred'
    trx_pct = f'TRx_{target_period}_PctError'
    display_cols = [
        'PRCSN_PAYER_ENT_NM', 'PRCSN_PBM_VENDOR' if is_payer_pbm else None, 'GRANULARITY_LEVEL', 'Anomaly_Priority',
        'Discriminator_Score', trx_actual, trx_pred, trx_pct,
        'Threshold_Score', 'Explanation',
        'SOP_Pass_Count', 'GAN_Anomaly_Confirmed_3of4', 'Novartis_Any_Flag', 'Novartis_Reason'
    ]
    display_cols = [c for c in display_cols if c and c in anomalies_df.columns]
    if not display_cols:
        display_cols = [c for c in ['PRCSN_PAYER_ENT_NM', 'GRANULARITY', 'Anomaly_Priority', 'Explanation'] if c in anomalies_df.columns]
    
    table_df = anomalies_df[display_cols].copy()
    
    if trx_pct in table_df.columns:
        table_df[trx_pct] = table_df[trx_pct].apply(lambda x: f"{float(x):.2f}%" if pd.notna(x) else "")
    
    # Add row numbers
    table_df.insert(0, 'Row', range(1, len(table_df) + 1))
    
    # Display table
    st.dataframe(
        table_df,
        use_container_width=True,
        height=400
    )

    # Export to Excel (uses full anomalies_df with all columns for download)
    try:
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            anomalies_df.to_excel(writer, sheet_name='Anomalies', index=False)
        excel_bytes = buffer.getvalue()
        st.download_button(
            label="Download as Excel",
            data=excel_bytes,
            file_name=f"Anomalies_{target_period}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )
    except ImportError:
        st.caption("Add openpyxl to requirements.txt for Excel export.")
    
    # Row selection dropdown
    st.markdown("### Select Anomaly for AI Analysis")
    
    # Reset index to ensure positional indexing works
    anomalies_df_reset = anomalies_df.reset_index(drop=True)
    
    # Create selection options
    anomaly_options = []
    for pos_idx in range(len(anomalies_df_reset)):
        row = anomalies_df_reset.iloc[pos_idx]
        entity = row.get('PRCSN_PAYER_ENT_NM', row.get('GRANULARITY', 'Unknown'))
        if is_payer_pbm and 'PRCSN_PBM_VENDOR' in row:
            entity = f"{entity} | {row.get('PRCSN_PBM_VENDOR', '')}"
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
    _is_pp = 'PRCSN_PBM_VENDOR' in selected_anomaly
    entity_name = selected_anomaly.get('PRCSN_PAYER_ENT_NM', selected_anomaly.get('GRANULARITY', 'Unknown'))
    if _is_pp:
        entity_name = f"{entity_name} | {selected_anomaly.get('PRCSN_PBM_VENDOR', '')}"
    granularity_level = selected_anomaly.get('GRANULARITY_LEVEL', 'Payer')
    
    # Time series for charts. Payer-PBM: T1-T18 only; Payer: T1-T25
    t_end = 18 if _is_pp else 25
    trx_ts = []
    nbrx_ts = []
    nrx_ts = []
    for i in range(1, t_end + 1):
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
    
    _tp = 'T18' if _is_pp else 'T24'
    _trx_actual = selected_anomaly.get(f'TRx_{_tp}_actual', selected_anomaly.get(f'TRx_{_tp}', 0))
    _trx_pred = selected_anomaly.get(f'TRx_{_tp}_pred', 0)
    _trx_pct = selected_anomaly.get(f'TRx_{_tp}_PctError', selected_anomaly.get('T24_to_T25_PctChange', 0))
    anomaly_data = {
        'Entity': entity_name,
        'Granularity_Level': granularity_level,
        'Anomaly_Score': selected_anomaly.get('Threshold_Score', selected_anomaly.get('Anomaly_Score', 0)),
        'Anomaly_Priority': selected_anomaly.get('Anomaly_Priority', selected_anomaly.get('Priority', 'Unknown')),
        'T24_to_T25_Change': _trx_pct,
        'TRx_T24': _trx_actual,
        'TRx_T25': selected_anomaly.get('TRx_T25', selected_anomaly.get('T25_TRx', 0)) if not _is_pp else _trx_pred,
        'Volatility_Category': selected_anomaly.get('TRx_Volatility_Category', 'Unknown'),
        'Trend_Direction': selected_anomaly.get('TRx_Trend_Direction', 'Unknown'),
        'Anomaly_Explanation': selected_anomaly.get('Explanation', selected_anomaly.get('Anomaly_Explanation', 'No explanation')),
    }
    
    # Build full row as string for LLM. For payer-PBM: only T1-T18 data (all records).
    def _row_str(d, max_len=250, t1_t18_only=False):
        lines = []
        for k, v in d.items():
            if pd.isna(v) or v == '':
                continue
            if t1_t18_only:
                m = re.match(r'^(\w+)_T(\d+)(.*)$', str(k))
                if m:
                    t = int(m.group(2))
                    if t < 1 or t > 18:
                        continue
            try:
                s = f"{k}: {v:.4g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}: {v}"
            except Exception:
                s = f"{k}: {v}"
            if len(s) > max_len:
                s = s[:max_len] + "..."
            lines.append(s)
        return "\n".join(lines)

    full_row_str = _row_str(selected_anomaly, t1_t18_only=_is_pp)

    # Build context for LLM based on dataset
    if _is_pp:
        vol_group = selected_anomaly.get('Volume_Group', 'GROUP1_High')
        dataset_context = f"DATASET: ML_DATASET_PAYER_PBM_T17_GROUP_{vol_group}.csv. High-volume payer-PBMs (deciles 1-3 for High, 4-6 for Mid, 7-10 for Low). This entity is in {vol_group}. We trained GAN on T1-T17, predicted T18, and flagged anomalies where actual T18 deviated from expected."
        time_context = "TIME PERIODS (use these so trend plots align): T1 = May 2023, T2 = Jun 2023, T3 = Jul 2023, T4 = Aug 2023, T5 = Sep 2023, T6 = Oct 2023, T7 = Nov 2023, T8 = Dec 2023, T9 = Jan 2024, T10 = Feb 2024, T11 = Mar 2024, T12 = Apr 2024, T13 = May 2024, T14 = Jun 2024, T15 = Jul 2024, T16 = Aug 2024, T17 = Sep 2024, T18 = Oct 2024. The data you receive is T1-T18 only. Use month names (May 2023 through Oct 2024) in your response so the trend plots below are coherent."
        anomaly_context = "ANOMALY: We identified an anomaly at TRx level for T18 (Oct 2024). Your job: drill down using relationships (NBRx/TRx, formulary, HCPs, states, MoM), all features, and find reasons why TRx dropped. Use Geoff-style language: strategic, not number dumps."
    else:
        dataset_context = "DATASET: Payer-level anomaly list. We compare predicted vs actual TRx at T24 (Apr 2025)."
        time_context = "TIME PERIODS: T1 = May 2023. T24 = Apr 2025 (anomaly target). Use month names in your response."
        anomaly_context = "ANOMALY: We identified an anomaly at TRx level for T24. Drill down using relationships and all features. Use Geoff-style language."

    # Call LLM for analysis
    if client is None:
        st.warning("⚠️ Azure OpenAI client not available. Using default analysis.")
        title = f"Anomaly: {entity_name}"
        key_insight = anomaly_data['Anomaly_Explanation']
        summary = f"This entity shows an anomaly with score {anomaly_data['Anomaly_Score']}. {anomaly_data['Anomaly_Explanation']}"
        key_drivers = ""
        key_relationship_violations = ""
        key_points = ""
    else:
        with st.spinner("🤖 AI is analyzing the anomaly..."):
            prompt_template = get_prompt()
            try:
                prompt = prompt_template.format(
                    dataset_context=dataset_context,
                    time_context=time_context,
                    anomaly_context=anomaly_context,
                    full_row_str=full_row_str
                )
            except KeyError:
                prompt = prompt_template.replace("{dataset_context}", dataset_context).replace("{time_context}", time_context).replace("{anomaly_context}", anomaly_context).replace("{full_row_str}", full_row_str)
            try:
                response = client.chat.completions.create(
                    model="ciathena-gpt-4o",
                    messages=[
                        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2500
                )
                
                llm_response = (response.choices[0].message.content or "").strip()
                
                # Parse LLM response (TITLE, ONE-LINER, KEY INSIGHT, SUMMARY, KEY DRIVERS, KEY POINTS)
                title = "Anomaly Analysis"
                key_insight = "Analysis in progress..."
                summary = "Processing..."
                key_drivers = ""
                key_relationship_violations = ""
                key_points = ""
                one_liner = ""

                if "TITLE:" in llm_response:
                    title = llm_response.split("TITLE:")[1].split("\n")[0].strip()
                    rest = llm_response.split("TITLE:")[1]
                    if "ONE-LINER:" in rest:
                        one_liner = rest.split("ONE-LINER:")[1].split("KEY INSIGHT:")[0].strip()
                    if "KEY INSIGHT:" in rest:
                        key_insight = rest.split("KEY INSIGHT:")[1].split("SUMMARY:")[0].strip()
                    if "SUMMARY:" in rest:
                        if "KEY DRIVERS:" in rest:
                            summary = rest.split("SUMMARY:")[1].split("KEY DRIVERS:")[0].strip()
                            drivers_block = rest.split("KEY DRIVERS:")[1]
                            if "KEY RELATIONSHIP VIOLATIONS:" in drivers_block:
                                key_drivers = drivers_block.split("KEY RELATIONSHIP VIOLATIONS:")[0].strip()
                                viol_block = drivers_block.split("KEY RELATIONSHIP VIOLATIONS:")[1]
                                if "KEY POINTS:" in viol_block:
                                    key_relationship_violations = viol_block.split("KEY POINTS:")[0].strip()
                                    key_points = viol_block.split("KEY POINTS:")[1].strip()
                                else:
                                    key_relationship_violations = viol_block.strip()
                            elif "KEY POINTS:" in drivers_block:
                                key_drivers = drivers_block.split("KEY POINTS:")[0].strip()
                                key_points = drivers_block.split("KEY POINTS:")[1].strip()
                            else:
                                key_drivers = drivers_block.strip()
                        elif "KEY POINTS:" in rest:
                            summary = rest.split("SUMMARY:")[1].split("KEY POINTS:")[0].strip()
                            key_points = rest.split("KEY POINTS:")[1].strip()
                        else:
                            summary = rest.split("SUMMARY:")[1].strip()
                elif "TITLE:" in llm_response and "KEY INSIGHT:" in llm_response:
                    key_insight = llm_response.split("KEY INSIGHT:")[1].split("SUMMARY:")[0].strip()
                    if "SUMMARY:" in llm_response and "KEY POINTS:" in llm_response:
                        summary = llm_response.split("SUMMARY:")[1].split("KEY POINTS:")[0].strip()
                        key_points = llm_response.split("KEY POINTS:")[1].strip()
                    key_relationship_violations = ""
                else:
                    # Custom prompt format (e.g. Opening paragraph, Why this matters now, Bottom line): show full response
                    title = f"Anomaly: {entity_name}"
                    key_insight = llm_response if llm_response else anomaly_data['Anomaly_Explanation']
                    summary = ""
                    key_drivers = ""
                    key_relationship_violations = ""
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
                key_drivers = ""
                key_relationship_violations = ""
                key_points = ""
    
    # Display LLM Analysis (remove markdown formatting from title)
    clean_title = title.replace('**', '').replace('*', '').strip()
    st.subheader(f"AI-Powered Anomaly Analysis: {clean_title}")

    # Display with section headers
    if one_liner:
        st.markdown("**One-Liner**")
        st.info(one_liner)
    if key_insight:
        st.markdown("**Key Insight**")
        st.info(key_insight)
    if summary:
        st.markdown("**Summary**")
        st.markdown(summary)
    if key_drivers:
        st.markdown("**Key Drivers**")
        st.markdown(key_drivers)
    if key_relationship_violations:
        st.markdown("**Key Relationship Violations**")
        st.markdown(key_relationship_violations)
    if key_points:
        st.markdown("**Key Points**")
        st.markdown(key_points)
    
    # Trend Lines
    trend_title = "May 2023 to Oct 2024 Trend Analysis" if _is_pp else "May 2023 to May 2025 Trend Analysis"
    st.markdown(f"### {trend_title}")
    
    # Create subplots for each metric
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('TRx Trend', 'NBRx Trend', 'NRx Trend'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Use actual month names in chronological order (oldest to newest, left to right)
    # Payer-PBM: T1-T18 (May 2023 to Oct 2024). Payer: T1-T25 (May 2023 to May 2025)
    time_periods = MONTH_NAMES[:t_end]  # T1 to T18 or T25
    trx_ts_chrono = trx_ts
    nbrx_ts_chrono = nbrx_ts
    nrx_ts_chrono = nrx_ts
    
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
    
    # Highlight anomaly period (T18 Oct 2024 for payer-PBM, T25 May 2025 for payer)
    anomaly_month = 'Oct 2024' if _is_pp else 'May 2025'
    anomaly_idx = (t_end - 1)  # T18 index 17 or T25 index 24
    if anomaly_idx < len(trx_ts_chrono):
        fig.add_trace(
            go.Scatter(
                x=[anomaly_month],
                y=[trx_ts_chrono[anomaly_idx]],
                mode='markers',
                name=f'Anomaly Period ({anomaly_month})',
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
        title_text=trend_title,
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
    st.markdown("### Anomaly Details")

    t24_t25_val = anomaly_data['T24_to_T25_Change']
    try:
        t24_t25_str = f"{float(t24_t25_val):.2f}%" if pd.notna(t24_t25_val) else "N/A"
    except (ValueError, TypeError):
        t24_t25_str = str(t24_t25_val)
    trx_label = f'TRx {target_period} % Error'
    trx_actual_label = f'TRx {target_period} actual'
    trx_pred_label = f'TRx {target_period} pred' if _is_pp else 'TRx T25'
    details_data = {
        'Metric': ['Entity', 'Granularity Level', 'Threshold Score', 'Priority',
                   trx_label, trx_actual_label, trx_pred_label, 'Volatility', 'Trend'],
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

