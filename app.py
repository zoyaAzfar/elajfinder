# importing libraries
import streamlit as st
import os
import re
import sqlite3
import json
import pandas as pd
import requests
from geopy.distance import geodesic
import ssl
from rapidfuzz import fuzz, process
from google import genai
from google.genai import types
import googlemaps 
import time
from google.genai.errors import APIError
from PIL import Image 
im = Image.open('favicon.png') 


st.set_page_config(     
    page_title="ElajFinder",     
    page_icon=im,     
    initial_sidebar_state="expanded" )

# reading csv file and converting it to a database
CSV_FILE = "NAME_FIXED_HOSPITALS.csv"
DB_FILE = "database.db"

df = pd.read_csv(CSV_FILE)
conn = sqlite3.connect(DB_FILE)
df.to_sql("data_table", conn, if_exists="replace", index=False)

# getting list of all departments to display on website
raw_depts = (
    df["Department"]
    .dropna()
    .astype(str)
    .str.split(",")
    .explode()
    .str.strip()
)

# keep only non-empty, valid string entries
ALL_DEPARTMENTS = sorted([
    dept for dept in raw_depts.unique()
    if dept and dept.lower() not in ["", "nan", "none", "null"]
])

# pre-fetch existing hospital names for fuzzy matching
EXISTING_HOSPITALS = df["Hospital Name"].dropna().unique().tolist()
conn.close()

STOPWORDS = {"hospital", "medical", "complex", "building", "old", "new", "the", "of", "and"}

# user table for saving history
def init_user_tables():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            user_key TEXT PRIMARY KEY,
            chat_history TEXT,
            saved_locations TEXT
        )
    """)
    conn.commit()
    conn.close()

init_user_tables()

def load_user_data(user_key):
    conn = sqlite3.connect(DB_FILE)
    row = conn.execute(
        "SELECT chat_history, saved_locations FROM user_data WHERE user_key=?",
        (user_key,)
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row[0] or "[]"), json.loads(row[1] or "{}")
    return [], {}

def save_user_data(user_key, history, locations):
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        INSERT INTO user_data (user_key, chat_history, saved_locations)
        VALUES (?, ?, ?)
        ON CONFLICT(user_key) DO UPDATE SET
            chat_history=excluded.chat_history,
            saved_locations=excluded.saved_locations
    """, (user_key, json.dumps(history), json.dumps(locations)))
    conn.commit()
    conn.close()



# this function builds a list of alias names for the actual names written in the database 
def build_alias_index(names):
    alias_map = {}
    first_word_counts = {}

    for name in names:
        alias_map[name.lower().strip()] = name

        # paranthetic abbreviations, e.g. "(IMC)" -> imc
        for paren in re.findall(r'\(([^)]+)\)', name):
            alias_map[paren.strip().lower()] = name

        # strip stopwords, use first 2 meaningful tokens as a short alias
        core = re.sub(r'\([^)]*\)', '', name)
        tokens = [t.strip(",.") for t in core.split() if t.lower() not in STOPWORDS]
        if tokens:
            if len(tokens) >= 2:
                alias_map[" ".join(tokens[:2]).lower()] = name
            first_word_counts.setdefault(tokens[0].lower(), []).append(name)

    # only alias a bare first word if it's unique across all hospitals
    for word, matches in first_word_counts.items():
        if len(matches) == 1:
            alias_map.setdefault(word, matches[0])

    return alias_map

ALIAS_INDEX = build_alias_index(EXISTING_HOSPITALS)

# tokenize hospita names
def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

# get significant tokens of hospital name with filler wods stripped
def _core_tokens(name: str) -> list[str]:
    core = re.sub(r'\([^)]*\)', '', name)
    return [t for t in _tokenize(core) if t not in STOPWORDS]

HOSPITAL_CORE_TOKENS = {name: _core_tokens(name) for name in EXISTING_HOSPITALS}

QUESTION_FILLER = STOPWORDS | {"compare", "vs", "versus", "between", "or", "a", "an"}

QUERY_CLEAN_WORDS = STOPWORDS | {"memorial", "care", "clinic", "center", "centre"}

def clean_hospital_query(text: str) -> str:
    tokens = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in QUERY_CLEAN_WORDS]
    return " ".join(tokens)

# compares each hospital's bare distinctive token(s) against each word in the
# user's input so a one letter typo isn't drowned out by filler like 'hospital"

def _best_hospital_matches(user_text: str, score_cutoff: int = 78, limit: int = 5):

    q_words = [w for w in _tokenize(user_text) if w not in QUESTION_FILLER]
    if not q_words:
        return []

    scored = []
    for name, core_tokens in HOSPITAL_CORE_TOKENS.items():
        if not core_tokens:
            continue
        best = max(
            max((fuzz.ratio(tok, qw) for qw in q_words), default=0)
            for tok in core_tokens
        )
        if best >= score_cutoff:
            scored.append((best, name))

    scored.sort(key=lambda x: -x[0])
    return [name for _, name in scored[:limit]]

def detect_mentioned_hospitals(user_question: str, score_cutoff: int = 80, limit: int = 5):
    """Scans free text for ALL likely hospital mentions, not just the single best match."""
    question_lower = user_question.lower()
    found = []

    # 1. Alias hits first (exact nicknames/abbreviations are cheap and reliable)
    for alias, canonical in ALIAS_INDEX.items():
        if alias in question_lower and canonical not in found:
            found.append(canonical)

    # Clean the input text so generic words like "memorial" or "hospital" don't trigger false matches
    cleaned_question = clean_hospital_query(question_lower)

    # 2. Fuzzy matches on the CLEANED question (Notice score_cutoff raised to 80)
    matches = process.extract(
        cleaned_question, 
        EXISTING_HOSPITALS, 
        scorer=fuzz.partial_token_set_ratio,
        limit=limit, 
        score_cutoff=score_cutoff
    )

    for name, score, _ in matches:
        if name not in found:
            found.append(name)

    return found

def resolve_hospital_name(user_input_name: str, score_cutoff: int = 78) -> str:
    clean_input = re.sub(r'\bhospital\b', '', user_input_name.lower())
    clean_input = re.sub(r'\s+', ' ', clean_input).strip()

    if clean_input in ALIAS_INDEX:
        return ALIAS_INDEX[clean_input]
    if user_input_name.lower().strip() in ALIAS_INDEX:
        return ALIAS_INDEX[user_input_name.lower().strip()]

    matches = _best_hospital_matches(user_input_name, score_cutoff, limit=1)
    if matches:
        return matches[0]

    input_words = clean_input.split()
    if input_words:
        for db_name in EXISTING_HOSPITALS:
            if all(word in db_name.lower() for word in input_words):
                return db_name

    return user_input_name

# automatic retry within limits due to free tier 
def call_gemini_safe(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(4 * (attempt + 1))   
                    continue
            raise e

#  gemini set up
api_key = os.environ.get("GEMINI_API_KEY") or (st.secrets.get("GEMINI_API_KEY") if "GEMINI_API_KEY" in st.secrets else None)

if not api_key:
    st.error("Please set GEMINI_API_KEY in your environment variables or Streamlit secrets.")
    st.stop()

gemini_client = genai.Client(api_key=api_key)

# installing google maps
gmaps_key = os.environ.get("GMAPS_API_KEY") or (st.secrets.get("GMAPS_API_KEY") if "GMAPS_API_KEY" in st.secrets else None)

if gmaps_key:
    gmaps = googlemaps.Client(key=gmaps_key)
else:
    st.warning("Google Maps API key missing. Distance calculations will fall back to straight-line estimates.")
    gmaps = None

# calculating distance as fallback
def get_straight_line_distance(user_lat, user_lon, hosp_lat, hosp_lon) -> float:
    """Fast local calculation in km."""
    return round(geodesic((user_lat, user_lon), (hosp_lat, hosp_lon)).km, 2)

# distance from google maps distance matrix api
def get_gmaps_road_distance(user_lat, user_lon, hosp_lat, hosp_lon):

    if gmaps:
        try:
            origin = (user_lat, user_lon)
            destination = (hosp_lat, hosp_lon)

            result = gmaps.distance_matrix(origins=[origin], destinations=[destination], mode="driving")

            if result['status'] == 'OK':
                element = result['rows'][0]['elements'][0]
                if element['status'] == 'OK':

                    distance_km = round(element['distance']['value'] / 1000.0, 2)
                    duration_min = round(element['duration']['value'] / 60.0)
                    return distance_km, duration_min
        except Exception as e:
            pass

    est_km = get_straight_line_distance(user_lat, user_lon, hosp_lat, hosp_lon)
    return est_km, round(est_km * 3)

# returns read-only sql database connection
def get_read_only_connection():
    return sqlite3.connect(f"file:{DB_FILE}?mode=ro", uri=True)

CITY_AVG_NEG_RATIO = df["Negative Reviews Ratio (%)"].astype(float).mean()

# performance labels for each hospital depending on negative review ratio (%)
def get_performance_label(hospital_ratio: float, comparison_avg: float, buffer: float = 5.0) -> str:

    if pd.isna(hospital_ratio):
        return "N/A"

    if hospital_ratio > (comparison_avg + buffer):
        return "Below Average Performance (High Negative Reviews)"
    elif hospital_ratio < (comparison_avg - buffer):
        return "Above Average Performance (Low Negative Reviews)"
    else:
        return "Average Performance"

# add the distance from the user to the dataframe 
def add_distance_to_dataframe(df_results: pd.DataFrame, user_coords: tuple) -> pd.DataFrame:
    """Appends 'Driving Distance' and 'Est. Drive Time' columns, then sorts nearest-first."""
    if not user_coords or df_results.empty:
        return df_results

    user_lat, user_lon = user_coords
    distances = []
    durations = []
    sort_keys = []

    for _, row in df_results.iterrows():
        try:
            coords_str = row.get("Coordinates")

            if pd.isna(coords_str):
                hosp_name = row.get("Hospital Name")
                if hosp_name:
                    resolved = resolve_hospital_name(hosp_name)
                    match = df[df["Hospital Name"] == resolved]
                    if not match.empty:
                        coords_str = match.iloc[0].get("Coordinates")

            if pd.isna(coords_str) or not isinstance(coords_str, str):
                raise ValueError("Missing coordinates in database.")

            parts = coords_str.split(',')
            if len(parts) != 2:
                raise ValueError("Invalid coordinates format.")

            hosp_lat = float(parts[0].strip())
            hosp_lon = float(parts[1].strip())

            dist_km, dur_min = get_gmaps_road_distance(user_lat, user_lon, hosp_lat, hosp_lon)            
            distances.append(f"{dist_km} km")
            durations.append(f"~{dur_min} mins")
            sort_keys.append(dist_km)

        except (ValueError, KeyError, TypeError, IndexError, AttributeError):
            distances.append("N/A")
            durations.append("N/A")
            sort_keys.append(float("inf"))

    df_results["Driving Distance"] = distances
    df_results["Est. Drive Time"] = durations
    df_results["_sort_km"] = sort_keys

    df_results = df_results.sort_values("_sort_km").drop(columns=["_sort_km"]).reset_index(drop=True)

    return df_results

# check to make sure gemini's query to the database is strictly read-only.  
def is_query_safe(sql: str) -> bool:
    clean_sql = sql.strip().upper()
    if not (clean_sql.startswith("SELECT") or clean_sql.startswith("WITH")):
        return False

    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
    for word in forbidden:
        if re.search(r'\b' + word + r'\b', clean_sql):
            return False

    return True

# overrides whatever columns the llm picked in SELECT keeps its WHERE/ORDER/LIMIT logic intact
def force_select_star(sql: str) -> str:
    return re.sub(r'(?is)^SELECT\s+.*?\s+FROM', 'SELECT * FROM', sql.strip(), count=1)

# returns compressed list of columns to save tokens
def get_schema():
    columns_list = df.columns.tolist()
    return "Table Name: data_table\nColumns available:\n" + ", ".join(columns_list)

# limiting rows to save tokens
def sanitize_df_for_prompt(df: pd.DataFrame, max_rows: int = 5, max_cell_chars: int = 200) -> pd.DataFrame:
    if df.empty:
        return df

    trimmed_df = df.head(max_rows).copy()

    for col in trimmed_df.columns:
        if trimmed_df[col].dtype == "object":
            trimmed_df[col] = trimmed_df[col].astype(str).apply(
                lambda x: x[:max_cell_chars] + "..." if len(x) > max_cell_chars else x
            )

    return trimmed_df

# fetches full data rows using flexible string matching.
def fetch_hospitals_for_comparison(hospital_names: list) -> pd.DataFrame:
    if not hospital_names:
        return pd.DataFrame()

    conn = get_read_only_connection()

    resolved_names = [resolve_hospital_name(name) for name in hospital_names]

    conditions = " OR ".join(["LOWER(\"Hospital Name\") LIKE LOWER(?)" for _ in resolved_names])
    query = f"SELECT * FROM data_table WHERE {conditions}"

    params = [f"%{name}%" for name in resolved_names]
    results_df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return results_df

# triggers when user asks to compare two hospitals
def ask_comparison(hospital_names: list, user_question: str):
    df_comparison = fetch_hospitals_for_comparison(hospital_names)

    if df_comparison.empty:
        yield f"Could not find matching hospitals for: {', '.join(hospital_names)}\n"
        return

    user_coords = st.session_state.get("user_coords")
    user_location_name = st.session_state.get("user_location_name", "Not specified")

    if user_coords:
        df_comparison = add_distance_to_dataframe(df_comparison, user_coords)

    location_line = (
        f'\nUser\'s stated location: "{user_location_name}" (distances below are calculated from here).\n'
        if user_coords else
        "\nUser location: not provided. Do not mention distance.\n"
    )

    safe_df_comp = sanitize_df_for_prompt(df_comparison, max_rows=5, max_cell_chars=150)

    selected_ratios = pd.to_numeric(safe_df_comp.get("Negative Reviews Ratio (%)"), errors="coerce")
    subset_avg_ratio = selected_ratios.mean()

    records = safe_df_comp.to_dict(orient="records")
    for row in records:
        ratio = pd.to_numeric(row.get("Negative Reviews Ratio (%)"), errors="coerce")
        row["Performance vs Selected Group"] = get_performance_label(ratio, subset_avg_ratio)
        row["Performance vs City Benchmark"] = get_performance_label(ratio, CITY_AVG_NEG_RATIO)

    hospitals_json = json.dumps(records, indent=2)

    comparison_prompt = f"""
    User question: "{user_question}"
    {location_line}

    City-Wide Average Negative Review Ratio: {CITY_AVG_NEG_RATIO:.1f}%
    Group Average Negative Review Ratio: {subset_avg_ratio:.1f}%

    Hospital Data:
    {hospitals_json}

    INSTRUCTIONS:
    1. LANGUAGE MATCHING: Detect the language and script of the user's question (Urdu script, Roman Urdu, or English) and write your entire response strictly in that same language/script.
    2. DO NOT use tables or headings under any circumstances as they are overwhelming.
    3. Include information on "overall performance benchmark" using 'Performance vs City Benchmark' or 'Performance vs Selected Group'.
    4. Highlight which hospitals perform Above Average (better) vs Below Average (worse) in negative feedback, with concrete examples why.
    5. Factor in the user's location from the distance to the hospital. 
    6. State clearly that you provide hospital choice guidance, not medical advice.
    7. Keep answers short - under 300 words.
    8. REMEMBER: ANY CURRENCY IS IN PAKISTANI RUPEE. 
    9. When using quotes, format accordingly. Keep CONCISE and BRIEF unless otherwise specified.
    """

    summary_response = call_gemini_safe(
        gemini_client.models.generate_content_stream,
        model="gemini-3.5-flash-lite",
        contents=comparison_prompt,
        config=types.GenerateContentConfig(temperature=0.1)
    )

    for chunk in summary_response:
        if chunk.text:
            yield chunk.text

# triggers when user asks general question
def ask_chatbot_general(user_question: str):
    schema = get_schema()

    all_hospitals_str = ", ".join(EXISTING_HOSPITALS)

    detected_hospitals = detect_mentioned_hospitals(user_question)
    if detected_hospitals:
        names_list = ", ".join(f'"{h}"' for h in detected_hospitals)
        hospital_hint = (
            f'\nIMPORTANT: The user is very likely referring to these exact hospital(s): {names_list}. '
            f'Use these EXACT spellings in your SQL LIKE clauses, do not alter them.\n'
        )
    else:
        hospital_hint = ""

    sql_prompt = f"""
    You are an automated SQL generator for SQLite table 'data_table'.

    AVAILABLE HOSPITALS IN DATABASE:
    {all_hospitals_str}

    CITY BENCHMARKS & CONTEXT:
    - The City-Wide Average 'Negative Reviews Ratio (%)' is {CITY_AVG_NEG_RATIO:.1f}%.

    STRICT QUERY RULES:
    1. Output ONLY valid, executable SQLite code. Include `LIMIT 15` if it's an open-ended search query to prevent massive outputs.
    2. Do NOT wrap query in markdown code blocks.
    3. Use `CAST("Negative Reviews Ratio (%)" AS REAL)` when doing numeric calculations or filtering.
    4. LANGUAGE & SPELLING CORRECTION: If the user asks in Urdu script (e.g., "حمید لطیف"), Roman Urdu (e.g., "hamed latif"), or misspells a name, you MUST look at the 'AVAILABLE HOSPITALS' list above, find the correct English spelling, and use THAT exact spelling in your SQL query.
    5. ALWAYS use case-insensitive partial matching. If the hospital name has multiple words, split them using AND. 
    Example: Use `LOWER("Hospital Name") LIKE LOWER('%integrated%') AND LOWER("Hospital Name") LIKE LOWER('%medical%')` instead of a single long string.
    6. CRITICAL: ALWAYS include the columns "Hospital Name" and "Coordinates" in your SELECT statement. If you do not include them, the system cannot calculate distances.

    Schema:
    {schema}
    {hospital_hint}
    Question: {user_question}
    """

    sql_completion = call_gemini_safe(
        gemini_client.models.generate_content,
        model="gemini-3.5-flash-lite",
        contents=sql_prompt,
        config=types.GenerateContentConfig(temperature=0.1)
    )

    raw_sql = sql_completion.text or ""

    sql_query = raw_sql.strip().replace("```sql", "").replace("```", "").strip()

    if not is_query_safe(sql_query):
        yield "Safety Guardrail Triggered: Modifying queries are disallowed."
        return

    sql_query = force_select_star(sql_query)   

    try:
        conn = get_read_only_connection()
        results_df = pd.read_sql_query(sql_query, conn)
        conn.close()
    except Exception as e:
        yield f"Database error executing SQL: {e}"
        return

    safe_df = sanitize_df_for_prompt(results_df, max_rows=5, max_cell_chars=250)

    user_coords = st.session_state.get("user_coords")
    user_location_name = st.session_state.get("user_location_name", "Not specified")

    if user_coords:
        results_df = add_distance_to_dataframe(results_df, user_coords)
        results_df = results_df.head(5)   

    safe_df = sanitize_df_for_prompt(results_df, max_rows=5, max_cell_chars=150)
    df_string = safe_df.to_string()

    location_line = (
        f'\nUser\'s stated location: "{user_location_name}" (hospitals below are sorted nearest-first for this user).\n'
        if user_coords else
        "\nUser location: not provided. Do not mention distance or nearby hospitals.\n"
    )

    summary_prompt = f"""
    User question: "{user_question}"
    Executed SQL: {sql_query}
    {location_line}
    Query Results:
    {safe_df.to_string()}

    STRICT FORMATTING INSTRUCTIONS:
    1. LANGUAGE MATCHING: Detect the language and script of the user's question (Urdu script, Roman Urdu, or English) and write your entire response strictly in that same language/script.
    2. Provide a brief overview answering the question directly.
    3. Do NOT include generic advice, "next steps", or preparation lists. Keep it CONCISE and BRIEF unless asked.
    4. State clearly that you provide hospital choice advice, not medical advice. Ask if they need further help.
    5. Use clear language with statistics or relevant quotes. Do not use tables or headings. Remember, you are here to answer in a conversational tone 
    to someone who does not understand a lot of complex data. So, for each data point you mention, analyze what this means about the hospital.
        EXAMPLE: "31% of all reviews about Hospital X are negative. This means that the sentiment is usually positive."
    6. Read the reviews for each hospital carefully and incorporate them into your feedback. Also incorporate the user's location and distance from the hospital.
    7. REMEMBER: ANY CURRENCY IS IN PAKISTANI RUPEE. 
    8. When using quotes, format accordingly. 
    

    IMPORTANT DATA DICTIONARY:
    - "Major Theme: [category] (%)" or "Sub-Theme: [category] (%)" represents the PERCENTAGE OF USER REVIEWS UNDER 3 STARS that mention this topic, NOT the hospital's total 
    operational capacity or specialty breakdown.  Since users can mistakenly tag stars, sometimes they may contain positive feedback.
    - Example: A 'Gynecology (%)' of 60% simply means 60% of online reviewers discussed the gynecology department. To check if a hospital offers that service, check the department list before reviews.
    """

    summary_response = call_gemini_safe(
        gemini_client.models.generate_content_stream,
        model="gemini-3.5-flash-lite",
        contents=summary_prompt,
        config=types.GenerateContentConfig(temperature=0.7)
    )

    for chunk in summary_response:
        if chunk.text:
            yield chunk.text

# analyzed user intent and creates sql prompt to lookup 
def route_user_query(user_question: str) -> dict:

    detected_hospitals = detect_mentioned_hospitals(user_question)
    hint_line = (
        f'\nLikely hospital(s) mentioned: {", ".join(detected_hospitals)}\n'
        if detected_hospitals else ""
    )

    router_prompt = f"""
    Analyze the user question and determine if they want to compare specific hospitals.
    Only flag intent as "compare" if the user uses words such as "compare" or asks you for information
    about two hospitals. If the user only mentioned ONE hospital in their query, do not flag as compare. 
    Available Known Hospitals in Database (sample): {json.dumps(EXISTING_HOSPITALS[:20])}

    User Question: "{user_question}"
    {hint_line}

    Respond strictly in JSON format matching one of these two structures:
    1. If comparing specific hospitals:
       {{"intent": "compare", "hospitals": ["Hospital Name 1", "Hospital Name 2"]}}
    
    2. For general search or filtering questions (including questions about a single hospital):
       {{"intent": "general", "hospitals": []}}
    """

    try:
        completion = call_gemini_safe(
            gemini_client.models.generate_content,
            model="gemini-3.5-flash-lite",
            contents=router_prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        data = json.loads(completion.text)

        # Apply fuzzy correction to extracted hospital names
        if data.get("hospitals"):
            data["hospitals"] = [resolve_hospital_name(h) for h in data["hospitals"]]
        return data
    except Exception:
        return {"intent": "general", "hospitals": []}

# uses recent chat history to rewrite the user's question into a standalone query.
def contextualize_question(user_question: str, chat_history: list) -> str:
    if not chat_history:
        return user_question

    recent_history = chat_history[-4:]

    history_str = ""
    for msg in recent_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        content_snippet = str(msg['content'])[:200]
        history_str += f"{role}: {content_snippet}...\n"

    context_prompt = f"""
    Given the following conversation history and the user's latest question, rewrite the latest question into a clear, standalone question. 

    CRITICAL RULES:
    1. If the user asks to "compare them", "compare these two", or "which one is better", look at the history and replace those pronouns with the EXACT hospital names discussed.
       Example History: User asked about Fatima Memorial, Lahore Care and Hameed Latif.
       User Question: "compare these three"
       Rewritten: "Compare Fatima Memorial Hospital, Lahore Care Hospital and Hameed Latif Hospital"
    2. Do NOT answer the question. ONLY return the rewritten sentence.

    Conversation History:
    {history_str}

    Latest User Question: "{user_question}"
    
    Standalone Question:
    """

    try:
        completion = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=context_prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        return completion.text.strip()
    except Exception:
        return user_question

# extracts known hospital names mentioned in recent chat messages.
def extract_hospitals_from_history(chat_history: list) -> list:
    found_hospitals = []
    # Search history backwards
    for msg in reversed(chat_history):
        content = str(msg.get("content", ""))
        for hosp in EXISTING_HOSPITALS:
            if hosp.lower() in content.lower() and hosp not in found_hospitals:
                found_hospitals.append(hosp)
            if len(found_hospitals) >= 2:
                return found_hospitals
    return found_hospitals

# debug mode commented out for deployment  st.sidebar.checkbox("Debug mode")
DEBUG = False

# process querry using helper functions above + adds debug mode to see behind the scenes
def process_query(user_question: str, chat_history: list):
    question_lower = user_question.lower()
    context_words = ["them", "these", "those", "both", "which one", "it", "the first", "the second"]
    needs_context = bool(chat_history) and any(
        re.search(r'\b' + re.escape(w) + r'\b', question_lower) for w in context_words
    )

    if needs_context:
        standalone_question = contextualize_question(user_question, chat_history)
    else:
        standalone_question = user_question

    detected_now = detect_mentioned_hospitals(standalone_question)

    wants_compare = bool(
        re.search(r"\b(compar|vs\.?|versus|better than|difference between)\b", standalone_question, re.I)
    ) or len(detected_now) >= 2

    hospitals = detected_now.copy()

    if wants_compare and len(hospitals) < 2:
        history_hospitals = extract_hospitals_from_history(chat_history)
        for h in history_hospitals:
            if h not in hospitals:
                hospitals.append(h)

    if DEBUG:
        with st.expander("Debug mode", expanded=True):
            st.write("LLM Contextualization Triggered:", needs_context)
            st.write("Standalone question:", standalone_question)
            st.write("Deterministic detection:", detected_now)
            st.write("Wants compare intent:", wants_compare)
            st.write("Final hospitals used:", hospitals)

    if wants_compare and len(hospitals) >= 2:
        return ask_comparison(hospitals[:2], standalone_question)

    return ask_chatbot_general(standalone_question)


# styling for page
page_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');

html, body, [class*="css"], .stApp, 
div[data-testid="stMarkdownContainer"] *, 
div[data-testid="stHeading"] * {
    font-family: "Inter", serif !important;
}

h1, h2, h3, h4,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stHeading"] {
    font-family: "Inter", serif !important;
    color: #1e293b !important;
    font-weight: 700 !important;
}

div[data-testid="stChatMessageAvatarUser"] {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border-radius: 50% !important;
}

div[data-testid="stChatMessageAvatarUser"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}

div[data-testid="stChatMessageAvatarAssistant"] {
    background-color: #e11d48 !important;
    color: #ffffff !important;
    border-radius: 50% !important;
}

div[data-testid="stChatMessageAvatarAssistant"] svg {
    fill: #ffffff !important;
    color: #ffffff !important;
}
</style>
"""

st.markdown(page_css, unsafe_allow_html=True)


st.title("Lahore Hospital Finder")
st.write("I can help you find information on and compare hospitals in Lahore!")

if "user_key" not in st.session_state:
    st.session_state.user_key = st.query_params.get("uid", None)

if not st.session_state.user_key:
    entered = st.sidebar.text_input("Enter a nickname to save your session:")
    if entered:
        st.session_state.user_key = entered
        st.query_params["uid"] = entered
        st.session_state.messages, st.session_state.saved_locations = load_user_data(entered)
        st.rerun()
elif "messages" not in st.session_state:
    st.session_state.messages, st.session_state.saved_locations = load_user_data(st.session_state.user_key)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# creating sidebar
with st.sidebar:

    # location picking 
    st.header("Enter Your Location")
    st.write("This allows the chatbot to factor in drive times and distance in its responses.") 
    user_location_str = st.text_input("Enter your neighborhood/area:", placeholder="e.g. DHA Phase 5...")

    if "user_coords" not in st.session_state:
        st.session_state.user_coords = None

    if user_location_str and st.button("Set Location"):
        if not gmaps:
            st.error("Google Maps is not configured. Please add your API key.")
        else:
            search_query = f"{user_location_str}, Lahore, Pakistan"

            with st.spinner("Finding location using Google Maps..."):
                try:
                    geocode_result = gmaps.geocode(search_query)

                    if geocode_result:
                        best_match = geocode_result[0]
                        lat = best_match['geometry']['location']['lat']
                        lng = best_match['geometry']['location']['lng']
                        formatted_address = best_match['formatted_address']

                        st.session_state.user_coords = (lat, lng)
                        st.session_state.user_location_name = user_location_str  

                        st.success(f"Location set: {formatted_address}")
                    else:
                        st.error("Google Maps couldn't find that location. Please try a different area name.")

                except Exception as e:
                    st.error(f"An unexpected error occurred with Google Maps: {e}")

    if st.session_state.get("saved_locations"):
        pick = st.sidebar.selectbox("Load a saved place:", [""] + list(st.session_state.saved_locations.keys()))
        if pick:
            st.session_state.user_coords = tuple(st.session_state.saved_locations[pick])
            st.session_state.user_location_name = pick

    save_label = st.sidebar.text_input("Save current location as:", placeholder="Home, Work...")
    if st.sidebar.button("Save Location") and st.session_state.get("user_coords") and save_label:
        st.session_state.saved_locations[save_label] = list(st.session_state.user_coords)
        save_user_data(st.session_state.user_key, st.session_state.messages, st.session_state.saved_locations)

    st.divider()

    # search for specific department or features
    st.subheader("Search for Specific Features")
    selected_dept = st.selectbox("Needs Department:", [""] + ALL_DEPARTMENTS)
    selected_sector = st.selectbox("Sector:", ["", "Private", "Public"])
    min_price, max_price = st.slider(
            "Price Range (PKR):",
            min_value=0,
            max_value=7000,
            value=(0, 7000),
            step=1000,
            format="PKR %d"
    )    
    sehat_card = st.checkbox("Accepts Sehat Sahulat Card")
    good_reviews = st.checkbox("Mostly Positive Reviews")
    large_hospital = st.checkbox("Large Capacity (Many Beds)")

    if st.button("Find Matching Hospitals"):
        prompt_parts = ["Find me hospitals"]

        if selected_sector:
            prompt_parts.append(f"in the {selected_sector} sector")
        if selected_dept:
            prompt_parts.append(f"that offer {selected_dept}")
        if min_price > 0 or max_price < 7000:
            if min_price == max_price:
                if min_price == 0:
                    prompt_parts.append("where the consultation/fee is free (PKR 0)")
                else:
                    prompt_parts.append(f"where the price is exactly PKR {min_price:,}")
            elif min_price == 0:
                prompt_parts.append(f"where the price is PKR {max_price:,} or less")
            elif max_price == 7000:
                prompt_parts.append(f"where the price is at least PKR {min_price:,}")
            else:
                prompt_parts.append(f"where the price is between PKR {min_price:,} and PKR {max_price:,}")
        if sehat_card:
            prompt_parts.append("that accept the Sehat Sahulat Card")
        if large_hospital:
            prompt_parts.append("with a large number of beds")
        if good_reviews:
            prompt_parts.append("where the negative review ratio is strictly below 50%")

        if len(prompt_parts) > 1:
            final_prompt = " ".join(prompt_parts) + "."
            st.session_state.preset_prompt = final_prompt
        else:
            st.warning("Please select at least one feature to search!")

    st.divider()

    # learn more about a singular hospital 
    st.subheader("Learn More About A Hospital")
    single_hosp = st.selectbox("Select a hospital:", [""] + EXISTING_HOSPITALS, key="single_hosp")
    if st.button("Ask about this hospital") and single_hosp:
        st.session_state.preset_prompt = f"Tell me about {single_hosp}, and what the reviews are like."

    st.divider()

    # comapre two hospitals
    st.subheader("Compare Hospitals")
    comp_a = st.selectbox("First Hospital:", [""] + EXISTING_HOSPITALS, key="comp_a")
    comp_b = st.selectbox("Second Hospital:", [""] + EXISTING_HOSPITALS, key="comp_b")
    if st.button("Go") and comp_a and comp_b:
        st.session_state.preset_prompt = f"Compare {comp_a} and {comp_b}"

# search bar
user_typed = st.chat_input("Ask me about hospitals in Lahore...")

prompt = user_typed or st.session_state.preset_prompt

if prompt:
    st.session_state.preset_prompt = None

    st.chat_message("user").markdown(prompt)

    current_history = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_generator = process_query(prompt, current_history)
        full_response = st.write_stream(response_generator)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    if st.session_state.get("user_key"):
        save_user_data(st.session_state.user_key, st.session_state.messages, st.session_state.get("saved_locations", {}))
