"""
financial_report_agent.py
"""

import os
import io
import json
import tempfile
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from langchain.tools import Tool
from typing import Dict, List, Tuple
from langchain.chains import LLMChain
from datetime import datetime, timezone
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.agents import initialize_agent, AgentType, Tool as LC_Tool

# ---------- Configuration ----------
# Read configuration from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # must be set
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")  # replace with whichever LLM you have
# ---------- End configuration ----------

# Initialize LLM (adjust to your LangChain version / provider)
#llm = OpenAI(openai_api_key=OPENAI_API_KEY, model_name=MODEL_NAME, base_url=OPENAI_BASE_URL, temperature=0.0)
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    temperature=0.0,
)
#print("LLM configured:", llm.model, "via", llm.base_url)

# ---------- Tools: Data Fetcher ----------
def fetch_price_history_yfinance(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical price and basic financial info via yfinance.
    Returns a DataFrame with Date index and columns: Open, High, Low, Close, Volume, etc.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=period, auto_adjust=False)
    if hist.empty:
        raise ValueError(f"No data returned for {ticker}")
    hist = hist.reset_index()
    # Add ticker column and return
    hist["ticker"] = ticker.upper()
    return hist

def fetch_financials_yfinance(ticker: str) -> Dict:
    """
    Return key financial statements using current yfinance API
    (no deprecated 'earnings' fields).
    """
    t = yf.Ticker(ticker)
    try:
        quarterly_income = t.quarterly_income_stmt
        quarterly_balance = t.quarterly_balance_sheet
        quarterly_cash = t.quarterly_cashflow
    except Exception:
        quarterly_income, quarterly_balance, quarterly_cash = None, None, None

    # Convert to serializable dicts if available
    def safe_to_dict(obj):
        try:
            return obj.to_dict()
        except Exception:
            return {}

    return {
        "income_stmt": safe_to_dict(quarterly_income),
        "balance_sheet": safe_to_dict(quarterly_balance),
        "cashflow": safe_to_dict(quarterly_cash),
    }

def normalize_period(user_period: str) -> str:
    """
    Convert natural language time periods into valid yfinance periods.
    """
    if not user_period:
        return "1y"

    p = user_period.strip().lower()
    # direct valid values
    valid = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    if p in valid:
        return p

    # natural mappings
    if "q1" in p or "q2" in p or "q3" in p or "q4" in p or "quarter" in p:
        return "3mo"
    if "month" in p:
        return "1mo"
    if "year" in p or "annual" in p:
        return "1y"
    if "ytd" in p:
        return "ytd"
    return "1y"  # fallback default

# Wrap tools for LangChain agent use
tools_for_agent = [
    LC_Tool(
        name="fetch_price_history",
        func=lambda ticker, period="2y": fetch_price_history_yfinance(ticker, period).to_json(date_format="iso"),
        description="Fetch historical OHLCV data for a ticker. Returns JSON string of DataFrame."
    ),
    LC_Tool(
        name="fetch_financials",
        func=lambda ticker: json.dumps(fetch_financials_yfinance(ticker)),
        description="Fetch quarterly financial statements for a ticker. Returns JSON string."
    ),
]

# ---------- Helper: Analysis ----------
def compute_quarterly_metrics(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Example: compute returns and a simple moving average.
    Expects df_prices with 'Date' column.
    """
    df = df_prices.copy()
    if "Date" not in df.columns:
        raise ValueError("DataFrame must have 'Date' column.")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['close'] = df['Close']
    df['daily_return'] = df['close'].pct_change()
    df['30d_ma'] = df['close'].rolling(window=30, min_periods=5).mean()
    return df

def compute_key_ratios(financials_json: Dict) -> Dict[str, float]:
    """
    Minimal example: compute a handful of ratios if values available.
    This function must be expanded with robust parsing for real-world statements.
    """
    # This is placeholder — real parsing would handle dict structures from yfinance or EDGAR
    ratios = {}
    try:
        # Example: parse net income and revenue if present
        # We'll be defensive: check keys and shapes
        fin = financials_json.get('financials', {}) if isinstance(financials_json, dict) else {}
        # placeholder values
        ratios['gross_margin'] = None
        ratios['net_profit_margin'] = None
    except Exception:
        pass
    return ratios

# ---------- Helper: Visualization ----------
def make_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, out_path: str):
    plt.figure(figsize=(9, 5))
    plt.plot(df[x_col], df[y_col])
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------- LLM Prompts ----------
INPUT_PARSER_PROMPT = PromptTemplate(
    input_variables=["user_request"],
    template="""
You are an assistant that extracts structured parameters for a financial report.
Input: {user_request}

Return valid JSON with keys:
- tickers: list of tickers (strings)
- period: one of [1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max]
- metrics: list of metrics requested (e.g., revenue growth, pe ratio, margins)
- output_format: "markdown" or "pdf"
- extras: optional dict

Return ONLY JSON.
"""
)


WRITE_REPORT_PROMPT = PromptTemplate(
    input_variables=["context_json", "key_findings", "charts"],
    template="""
You are a senior financial analyst writing a concise report for analysts. Use the context and metrics below.
Context (JSON): {context_json}

Key findings (JSON): {key_findings}

Charts: {charts}

Write a clear Markdown report (2–4 pages). Include:
- Executive summary (2–4 sentences)
- Key metrics table
- Chart captions referencing filenames
- Earnings call sentiment summary (if provided), including sentiment classification and 2–3 direct quotes.
- Risks & opportunities (bullet points)
- Explainability Section: "Why I Concluded X" - enumerate exactly which numerical values or trends led to your conclusions.
Return only the markdown content.
"""
)

REVIEW_PROMPT = PromptTemplate(
    input_variables=["draft_markdown", "data_summary"],
    template="""
You are a reviewer whose job is to check the draft for:
1) Numeric consistency with the provided data_summary
2) Ambiguous or misleading claims
3) Tone / compliance language

Data summary (JSON): {data_summary}

Draft markdown:
{draft_markdown}

Return a JSON with fields:
- issues: list of strings describing issues (empty list if none)
- revised: revised markdown (if you made edits), else return original under 'revised'.
"""
)

# ---------- Chains ----------
input_parser_chain = INPUT_PARSER_PROMPT | llm
writer_chain = WRITE_REPORT_PROMPT | llm
review_chain = REVIEW_PROMPT | llm

# ---------- Orchestrator ----------
def orchestrator_run(user_request: str, output_dir: str = "./out") -> Dict:
    """
    Main orchestrator: parse -> fetch -> analyze -> visualize -> write -> review -> export
    Returns metadata including paths to artifacts and final markdown.
    """
    os.makedirs(output_dir, exist_ok=True)
    # 1) Parse input
    parsed = input_parser_chain.invoke({"user_request": user_request}).content
    try:
        parsed_json = json.loads(parsed)
    except Exception:
        # fallback: naive parse if chain returned lines; attempt to extract json
        parsed_json = {}
    # Defensive defaults
    tickers = parsed_json.get("tickers") or ["AAPL", "MSFT"]
    period = normalize_period(parsed_json.get("period"))
    metrics = parsed_json.get("metrics") or ["revenue growth", "P/E ratio", "net margin"]
    output_format = parsed_json.get("output_format") or "markdown"

    # Containers
    data_summary = {}
    chart_files = []

    # 2) For each ticker: fetch data and compute metrics
    for ticker in tickers:
        try:
            df_json = tools_for_agent[0].func(ticker, period)  # call fetch_price_history via wrapper
            df = pd.read_json(io.StringIO(df_json))
            # Compute simple metrics
            df_analysis = compute_quarterly_metrics(df)
            # Save CSV
            csv_path = os.path.join(output_dir, f"{ticker}_prices.csv")
            df_analysis.to_csv(csv_path, index=False)
            # Make chart
            chart_path = os.path.join(output_dir, f"{ticker}_close.png")
            make_line_chart(df_analysis, "Date", "close", f"{ticker} Close Price", chart_path)
            chart_files.append(chart_path)
            # Fetch financials
            fin_json_str = tools_for_agent[1].func(ticker)
            fin_json = json.loads(fin_json_str) if isinstance(fin_json_str, str) else fin_json_str
            ratios = compute_key_ratios(fin_json)
            # Summarize for reviewer
            data_summary[ticker] = {
                "csv": csv_path,
                "chart": chart_path,
                "ratios": ratios,
                "sample_price_rows": df_analysis.tail(3).to_dict(orient="records")
            }
        except Exception as e:
            data_summary[ticker] = {"error": str(e)}

    # 3) Draft report via writer LLM
    context_json = {
        "tickers": tickers,
        "period": period,
        "metrics": metrics
    }
    # key_findings = {"note": "Auto-generated; include computed metrics above."}
    key_findings = {
        "auto_metrics": {
            t: {
                "latest_close": float(df_analysis["close"].iloc[-1]),
                "avg_return": float(df_analysis["daily_return"].mean()),
                "sentiment": data_summary[t].get("sentiment", {}).get("sentiment"),
            }
            for t in tickers
        },
        "note": "Auto-generated using financial data and sentiment analysis."
    }
    charts_list = chart_files
    draft_markdown = writer_chain.invoke({
        "context_json": json.dumps(context_json),
        "key_findings": json.dumps(key_findings),
        "charts": json.dumps(charts_list)
    }).content

    # 4) Run reviewer
    review_result = review_chain.invoke({
        "draft_markdown": draft_markdown,
        "data_summary": json.dumps(data_summary)
    }).content
    
    try:
        review_json = json.loads(review_result)
    except Exception:
        review_json = {"issues": [], "revised": draft_markdown}

    final_markdown = review_json.get("revised", draft_markdown)

    # Save final markdown
    report_file = f"financial_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.md"
    final_md_path = os.path.join(output_dir, report_file)
    with open(final_md_path, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    # 5) Optionally convert to PDF using pypandoc/reportlab (not required here)
    result = {
        "final_markdown_path": final_md_path,
        "charts": chart_files,
        "data_summary": data_summary,
        "review_issues": review_json.get("issues", [])
    }
    return result

# ---------- If run as script ----------
if __name__ == "__main__":
    # Example user request: you can change it to any natural language instruction
    user_request = "Generate a Q2 2025 comparison report for AAPL and MSFT covering revenue growth, net margins, and P/E ratio. Output: markdown."
    print("Running orchestrator...")
    out = orchestrator_run(user_request, output_dir="./report_out")
    print("Done. Artifacts:")
    print(json.dumps(out, indent=2))
