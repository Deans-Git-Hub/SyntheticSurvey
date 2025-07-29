#!/usr/bin/env python3
import os
import json
import re
from collections import Counter
import difflib

import streamlit as st
import pandas as pd
import altair as alt
import openai


import streamlit as st

# ─── Session State ─────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="Secure App", layout="centered")

# ─── Custom CSS ────────────────────────────────────────────
st.markdown(
    """
    <style>
    .login-container {
        max-width: 400px;
        margin: auto;
        background: #f9f9f9;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    .login-header {
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .login-header img {
        width: 80px;
        margin-bottom: 0.5rem;
    }
    .login-header h2 {
        margin: 0;
        font-size: 1.5rem;
    }
    .stButton>button {
        width: 100%;
        padding: 0.7rem 0;
        font-size: 1rem;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Login Form ────────────────────────────────────────────
def login():
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="login-header">
                <!-- Optional logo -->
                <!--<img src="https://yourdomain.com/logo.png" alt="Logo">-->
                <h2>🔒 Secure Login</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login_form"):
            pwd = st.text_input(
                "Enter password",
                type="password",
                help="You’ll need this to view the app.",
                clear_on_submit=True,
            )
            submit = st.form_submit_button("Unlock")
            if submit:
                if pwd == st.secrets["credentials"]["password"]:
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("❌ Incorrect password")
        st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.authenticated:
    login()
    st.stop()

# ─── Protected Content ─────────────────────────────────────
st.success("✅ Access granted!")
st.title("Welcome to Your Secure Streamlit App")
st.write("Here’s the confidential content…")


# —— 1) Setup —— #
st.set_page_config(page_title="Synthetic Survey Engine", layout="wide")
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_chat(messages, fn=None, fn_name=None, temp=1.0):
    payload = {"model": "o4-mini", "messages": messages, "temperature": temp}
    if fn:
        payload["functions"] = [fn]
        payload["function_call"] = {"name": fn_name}
    return openai.chat.completions.create(**payload).choices[0].message

# —— 2) Sidebar controls —— #
st.sidebar.header("Survey Configuration")
industry   = st.sidebar.text_input("Industry name", value="Pepsi")
segment    = st.sidebar.text_input("Persona segment (optional)", value="Health Buffs")
n_personas = st.sidebar.number_input("Number of personas", min_value=5, max_value=200, value=10, step=5)
run_button = st.sidebar.button("Run survey")

# —— 3) Initialize session state —— #
if "persona_fields" not in st.session_state:
    st.session_state.persona_fields = [
        {"name": "name", "type": "string"},
        {"name": "age", "type": "integer"},
        {"name": "country", "type": "string"},
        {"name": "city", "type": "string"},
        {"name": "gender", "type": "string"},
        {"name": "marital_status", "type": "string"},
        {"name": "income", "type": "string"},
        {"name": "urbanicity", "type": "string"},
        {"name": "education", "type": "string"},
        {"name": "occupation", "type": "string"},
        {"name": "intro", "type": "string"},
    ]
if "questions" not in st.session_state:
    st.session_state.questions = [
        {
            "key": "alignment",
            "system": "Answer Yes or No exactly. Deeply consider the choices and choose the one that best aligns with you as a person.",
            "user": "Is Pepsi easily accessible?",
            "options": ["Yes", "No"]
        },
        {
            "key": "interest",
            "system": "Choose one option. Deeply consider the choices and choose the one that best aligns with you as a person.",
            "user": "At what time would you drink soda?",
            "options": ["Morning", "Afternoon", "Evening"]
        },
        {
            "key": "timeline",
            "system": "Choose one option. Deeply consider the choices and choose the one that best aligns with you as a person.",
            "user": "How often do you drink soda?",
            "options": ["Frequently", "Occasionally", "Rarely", "Never"]
        },
    ]

# —— Callbacks for add/remove —— #
def remove_persona_field(idx):
    st.session_state.persona_fields.pop(idx)

def add_persona_field():
    intro_idx = next((i for i, f in enumerate(st.session_state.persona_fields) if f["name"] == "intro"), None)
    new_field = {"name": "", "type": "string"}
    if intro_idx is not None:
        st.session_state.persona_fields.insert(intro_idx, new_field)
    else:
        st.session_state.persona_fields.append(new_field)

def remove_question(idx):
    st.session_state.questions.pop(idx)

def add_question():
    st.session_state.questions.append({"key": "", "system": "", "user": "", "options": []})

# —— 4) Layout with tabs —— #
tab_config, tab_results = st.tabs(["Configuration", "Results"])

with tab_config:
    st.header("Persona Attributes")
    for i, f in enumerate(st.session_state.persona_fields):
        if f["name"] == "intro":
            st.markdown("**intro** (fixed field)")
            continue
        c1, c2, c3 = st.columns([4, 2, 1])
        name = c1.text_input("Field name", f["name"], key=f"pf_name_{i}")
        typ  = c2.selectbox("Type", ["string", "integer"],
                            index=["string", "integer"].index(f["type"]),
                            key=f"pf_type_{i}")
        st.session_state.persona_fields[i] = {"name": name.strip(), "type": typ}
        c3.button("Remove", key=f"rm_pf_{i}", on_click=remove_persona_field, args=(i,))
    st.button("Add persona field", on_click=add_persona_field)

    st.header("Survey Questions")
    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Question {i+1}**")
        key  = st.text_input("Key", q["key"], key=f"q_key_{i}")
        sys  = st.text_input("System prompt", q["system"], key=f"q_sys_{i}")
        user = st.text_input("User prompt", q["user"], key=f"q_user_{i}")
        opts = st.text_input("Options (comma-separated)", ", ".join(q["options"]), key=f"q_opts_{i}")
        st.session_state.questions[i] = {
            "key": key.strip(),
            "system": sys.strip(),
            "user": user.strip(),
            "options": [o.strip() for o in opts.split(",") if o.strip()]
        }
        st.button("Remove question", key=f"rm_q_{i}", on_click=remove_question, args=(i,))
    st.button("Add survey question", on_click=add_question)

# —— 5) Schema & persona function —— #
def build_persona_schema(fields):
    props, req = {}, []
    for f in fields:
        props[f["name"]] = {"type": "integer"} if f["type"] == "integer" else {"type": "string"}
        req.append(f["name"])
    return {"type": "object", "properties": props, "required": req}

def make_persona_fn(schema):
    return {
        "name": "generate_personas",
        "description": "Generate customer personas.",
        "parameters": {
            "type": "object",
            "properties": {"personas": {"type": "array", "items": schema}},
            "required": ["personas"]
        }
    }

def generate_personas(segment, n, schema):
    fn = make_persona_fn(schema)
    seg_txt = f" They all belong to “{segment}”." if segment else ""
    sys_msg = {
        "role": "system",
        "content": (
            f"Generate up to {n} unique customer personas.{seg_txt} "
            "Each persona should reflect that background, but exhibit a wide spectrum "
            "of preferences and opinions, not uniformly positive or negative."
        )
    }
    personas, seen = [], set()
    while len(personas) < n:
        need = n - len(personas)
        resp = call_chat(
            [sys_msg, {"role": "user", "content": f"Generate {need} personas now."}],
            fn=fn, fn_name="generate_personas"
        )
        batch = json.loads(resp.function_call.arguments)["personas"]
        for p in batch:
            uid = p.get("intro", "") + p.get("name", "")
            if uid not in seen:
                personas.append(p)
                seen.add(uid)
            if len(personas) >= n:
                break
    return personas

# —— 6) Improved parser —— #
def parse_choice(txt, options, cutoff=0.8):
    text = (txt or "").lower()
    matches = []
    for opt in options:
        m = re.search(rf"\b{re.escape(opt.lower())}\b", text)
        if m:
            matches.append((opt, m.start()))
    if matches:
        return min(matches, key=lambda x: x[1])[0]
    fuzzy = difflib.get_close_matches(text, [o.lower() for o in options], n=1, cutoff=cutoff)
    if fuzzy:
        return next(o for o in options if o.lower() == fuzzy[0])
    return None

def run_survey(personas, questions, segment):
    scores = {q["key"]: [] for q in questions}
    for p in personas:
        lines = [f"{f['name'].capitalize()}: {p.get(f['name'], '')}" for f in st.session_state.persona_fields]
        if segment:
            lines.append(f"Segment: {segment}")
        base = {"role":"system","content":"You are this persona:\n" + "\n".join(lines)}
        for q in questions:
            msg = call_chat([base, {"role":"system","content":q["system"]}, {"role":"user","content":q["user"]}])
            choice = parse_choice(getattr(msg, "content", ""), q["options"])
            if choice is None:
                retry = call_chat(
                    [base, {"role":"system","content":"Answer with exactly one of: " + ", ".join(q["options"])}, {"role":"user","content":q["user"]}],
                    temp=1
                )
                choice = parse_choice(getattr(retry, "content", ""), q["options"]) or q["options"][-1]
            scores[q["key"]].append(choice)
    return scores

# —— 7) Results & Key Findings —— #
with tab_results:
    if run_button:
        schema    = build_persona_schema(st.session_state.persona_fields)
        questions = st.session_state.questions

        with st.spinner("Generating personas…"):
            personas = generate_personas(segment, n_personas, schema)
        with st.spinner("Running survey…"):
            scores = run_survey(personas, questions, segment)

        st.title("Synthetic Survey Results")
        total = len(personas)

        # Per-question charts & tables
        for q in questions:
            dist = Counter(scores[q["key"]])
            df = pd.DataFrame({
                "Option":  q["options"],
                "Count":   [dist.get(o,0) for o in q["options"]],
                "Percent": [round(dist.get(o,0)/total*100,1) for o in q["options"]],
            })
            st.header(q["user"])
            chart = (
                alt.Chart(df)
                   .mark_bar()
                   .encode(
                       x=alt.X("Option:N", sort=q["options"]),
                       y="Count:Q",
                       tooltip=["Option","Count","Percent"]
                   )
                   .properties(width=600, height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            st.table(df.set_index("Option"))

        st.header("Generated Personas")
        st.dataframe(pd.DataFrame(personas))

        st.header("Persona Intros")
        for p in personas:
            st.markdown(f"**{p.get('name','')}**: {p.get('intro','')}")

        # —— Key Findings —— #
        st.header("Key Findings")

        # Compute stats
        stats = {}
        for q in questions:
            key = q["key"]
            opts = q["options"]
            dist = Counter(scores[key])
            stats[key] = {
                "counts": {o: dist.get(o,0) for o in opts},
                "percentages": {o: round(dist.get(o,0)/total*100,1) for o in opts}
            }

        # —— AI‑generated summary —— #
        personas_json = json.dumps(personas, indent=2)
        stats_json    = json.dumps(stats, indent=2)
        find_prompt = [
            {
                "role":"system",
                "content":(
                    f"Synthetic Survey Engine results for industry '{industry}'"
                    f"{' (segment: '+segment+')' if segment else ''}.\n\n"
                    f"Metrics:\n{stats_json}\n\n"
                    f"Persona profiles:\n{personas_json}"
                )
            },
            {
                "role":"user",
                "content":(
                    "Based on these metrics and persona profiles, write several short paragraphs "
                    "highlighting overall response trends and which persona characteristics "
                    "most strongly correlate with particular answer patterns."
                )
            }
        ]
        find_fn = {
            "name":"generate_findings",
            "description":"Return an object with a 'summary' field containing multiple paragraphs of analysis.",
            "parameters":{
                "type":"object",
                "properties":{"summary":{"type":"string"}},
                "required":["summary"]
            }
        }
        find_resp = call_chat(find_prompt + [{"role":"system","content":"You are a data analyst."}],
                              fn=find_fn, fn_name="generate_findings", temp=1)
        summary = json.loads(find_resp.function_call.arguments)["summary"]
        for para in summary.split("\n\n"):
            st.write(para)
