#!/usr/bin/env python3
import os
import json
from collections import Counter, defaultdict

import streamlit as st
import pandas as pd
import altair as alt
import openai

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
segment    = st.sidebar.text_input(
    "Persona segment (optional)",
    value="Health Buffs",
    help="All synthetic respondents will fall under this segment."
)
n_personas = st.sidebar.number_input(
    "Number of personas",
    min_value=5, max_value=200, value=50, step=5
)
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
        {"key":"alignment","system":"Answer Yes or No exactly. Deeply consider the choices and choose the one that best aligns with you as a person.","user":"Is Pepsi easily accessible?","options":["Yes","No"]},
        {"key":"interest","system":"Choose one option. Deeply consider the choices and choose the one that best aligns with you as a person.","user":"At what time do you drink soda?","options":["Morning", "Afternoon", "Evening"]},
        {"key":"timeline","system":"Choose one option. Deeply consider the choices and choose the one that best aligns with you as a person.","user":"How often do you drink soda?","options":["Frequently","Occasionally","Rarely","Never"]},
    ]

# —— 4) Layout with tabs —— #
tab_config, tab_results = st.tabs(["Configuration", "Results"])

with tab_config:
    # Persona attributes
    st.header("Persona Attributes")
    for idx, field in enumerate(st.session_state.persona_fields):
        cols = st.columns([4, 2, 1])  # wider first column for alignment
        name = cols[0].text_input("Field name", field["name"], key=f"pf_name_{idx}")
        typ  = cols[1].selectbox("Type", ["string","integer"],
                                 index=["string","integer"].index(field["type"]),
                                 key=f"pf_type_{idx}")
        if cols[2].button("Remove", key=f"pf_remove_{idx}"):
            st.session_state.persona_fields.pop(idx)
        st.session_state.persona_fields[idx] = {"name": name.strip(), "type": typ}
    if st.button("Add persona field"):
        st.session_state.persona_fields.append({"name": "", "type": "string"})

    # Survey questions
    st.header("Survey Questions")
    for idx, q in enumerate(st.session_state.questions):
        st.markdown(f"**Question {idx+1}**")
        key  = st.text_input("Key", q["key"], key=f"q_key_{idx}")
        sys  = st.text_input("System prompt", q["system"], key=f"q_sys_{idx}")
        user = st.text_input("User prompt", q["user"], key=f"q_user_{idx}")
        opts = st.text_input("Options (comma-separated)", ", ".join(q["options"]), key=f"q_opts_{idx}")
        if st.button("Remove question", key=f"q_remove_{idx}"):
            st.session_state.questions.pop(idx)
        st.session_state.questions[idx] = {
            "key": key.strip(),
            "system": sys.strip(),
            "user": user.strip(),
            "options": [o.strip() for o in opts.split(",") if o.strip()]
        }
    if st.button("Add survey question"):
        st.session_state.questions.append({"key":"","system":"","user":"","options":[]})

# —— 5) Schema & persona function —— #
def build_persona_schema(fields):
    props, req = {}, []
    for f in fields:
        props[f["name"]] = {"type": "integer"} if f["type"]=="integer" else {"type":"string"}
        req.append(f["name"])
    return {"type":"object","properties":props,"required":req}

def make_persona_fn(schema):
    return {
        "name": "generate_personas",
        "description": "Generate customer personas.",
        "parameters": {"type":"object","properties":{"personas":{"type":"array","items":schema}},"required":["personas"]}
    }

def generate_personas(segment, n, schema):
    fn = make_persona_fn(schema)
    seg_txt = f" They all belong to the “{segment}” segment." if segment else ""
    sys_msg = {
        "role":"system",
        "content":(
            f"Generate up to {n} unique customer personas.{seg_txt} "
            "Each persona should reflect that background, but exhibit a wide spectrum "
            "of preferences and opinions, not uniformly positive or negative."
        )
    }
    personas, seen = [], set()
    while len(personas) < n:
        need     = n - len(personas)
        user_msg = {"role":"user","content":f"Generate {need} personas now."}
        resp     = call_chat([sys_msg, user_msg], fn=fn, fn_name="generate_personas")
        batch    = json.loads(resp.function_call.arguments)["personas"]
        for p in batch:
            uid = p.get("intro","") + p.get("name","")
            if uid not in seen:
                personas.append(p)
                seen.add(uid)
            if len(personas) >= n:
                break
    return personas

# —— 6) Survey runner —— #
def parse_choice(txt, options):
    t = (txt or "").strip().lower()
    for o in options:
        if o.lower() in t:
            return o
    return options[-1]

def run_survey(personas, questions, segment):
    scores = {q["key"]:[] for q in questions}
    for p in personas:
        lines = [f"{f['name'].capitalize()}: {p.get(f['name'], '')}" for f in st.session_state.persona_fields]
        if segment:
            lines.append(f"Segment: {segment}")
        base = {"role":"system","content":"You are this persona:\n"+ "\n".join(lines)}
        for q in questions:
            msg = call_chat([base, {"role":"system","content":q["system"]}, {"role":"user","content":q["user"]}])
            scores[q["key"]].append(parse_choice(getattr(msg,"content",""), q["options"]))
    return scores

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

        for q in questions:
            dist = Counter(scores[q["key"]])
            df   = pd.DataFrame({
                "Option":  q["options"],
                "Count":   [dist.get(o,0) for o in q["options"]],
                "Percent": [round(dist.get(o,0)/total*100,1) for o in q["options"]],
            })
            st.header(q["user"])
            chart = (
                alt.Chart(df)
                   .mark_bar()
                   .encode(x=alt.X("Option:N", sort=q["options"]),
                           y="Count:Q",
                           tooltip=["Option","Count","Percent"])
                   .properties(width=600, height=300)
            )
            st.altair_chart(chart, use_container_width=True)
            st.table(df.set_index("Option"))

        st.header("Generated Personas")
        st.dataframe(pd.DataFrame(personas))

        st.header("Persona Intros")
        for p in personas:
            st.markdown(f"**{p.get('name','')}**: {p.get('intro','')}")
