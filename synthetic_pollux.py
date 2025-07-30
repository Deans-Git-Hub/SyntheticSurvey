import os
import json
from collections import Counter

import streamlit as st
import pandas as pd
import altair as alt
import openai

st.set_page_config(page_title="Protected App", layout="wide")

# 1) Load password from secrets
PASSWORD = st.secrets.get("password")
if PASSWORD is None:
    st.error(
        "⚠️ No `password` in secrets!\n\n"
        "Add in `.streamlit/secrets.toml`:\n\n"
        "    password = \"Synthetic!\"\n\n"
        "or set it in your Streamlit Cloud Secrets."
    )
    st.stop()

# 2) Init “logged in” flag
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# 3) Show *only* the login form if not already authenticated
if not st.session_state.authenticated:
    st.title("🔐 Please log in")

    with st.form("login_form"):
        pw     = st.text_input("Password", type="password", placeholder="••••••")
        submit = st.form_submit_button("Unlock")

    if submit:
        if pw == PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password.")

    st.stop()

# 4) PROTECTED CONTENT
st.title("Welcome to SurveySynth!")

# —— 1) Setup —— #
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
n_personas = st.sidebar.number_input("Number of personas", min_value=5, max_value=50, value=10, step=5)
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
            "key": "accessibility",
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

# Callbacks for add/remove (omitted here for brevity)...

# —— 4) Layout with tabs —— #
tab_config, tab_results = st.tabs(["Configuration", "Results"])

with tab_config:
    st.header("Persona Attributes")
    # ... existing persona field UI code ...
    st.header("Survey Questions")
    # ... existing survey question UI code ...

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

# —— New: Batch function schema —— #
def make_batch_fn(questions):
    return {
        "name": "answer_survey",
        "description": "Answer multiple survey questions at once.",
        "parameters": {
            "type": "object",
            "properties": {
                **{q["key"]: {"type": "string", "enum": q["options"]} for q in questions}
            },
            "required": [q["key"] for q in questions]
        }
    }

# —— 6) Batched survey runner —— #
def run_survey(personas, questions, segment):
    scores = {q["key"]: [] for q in questions}
    batch_fn = make_batch_fn(questions)

    for p in personas:
        # Build persona context
        lines = [f"{f['name'].capitalize()}: {p.get(f['name'], '')}" for f in st.session_state.persona_fields]
        if segment:
            lines.append(f"Segment: {segment}")
        base = {
            "role": "system",
            "content": "You are this persona:\n" + "\n".join(lines)
        }

        # Build message list: base + all system prompts
        messages = [base]
        for q in questions:
            messages.append({"role": "system", "content": q["system"]})

        # Combine all user questions into one content block
        user_text = "\n".join(f"{i+1}. {q['user']}" for i, q in enumerate(questions))
        messages.append({"role": "user", "content": user_text})

        # Single API call per persona
        resp = call_chat(messages, fn=batch_fn, fn_name="answer_survey")
        answers = json.loads(resp.function_call.arguments)

        # Distribute answers into scores
        for q in questions:
            scores[q["key"]].append(answers.get(q["key"]))

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
        stats = {}
        for q in questions:
            key = q["key"]
            opts = q["options"]
            dist = Counter(scores[key])
            stats[key] = {
                "counts": {o: dist.get(o,0) for o in opts},
                "percentages": {o: round(dist.get(o,0)/total*100,1) for o in opts}
            }

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
