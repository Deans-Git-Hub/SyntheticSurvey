import os
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import pandas as pd
import altair as alt
import openai

# ——— 0) Page config & API key —————————————————————
st.set_page_config(page_title="Synthetic Survey Engine", layout="wide")
openai.api_key = os.getenv("OPENAI_API_KEY")

# ——— 1) Password gate ——————————————————————————
PASSWORD = st.secrets.get("password")
if PASSWORD is None:
    st.error(
        "⚠️ No `password` in secrets!\n\n"
        "Add in `.streamlit/secrets.toml`:\n\n"
        "    password = \"Synthetic!\"\n\n"
        "or set it in your Streamlit Cloud Secrets."
    )
    st.stop()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

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

# ——— 2) Initialize session state defaults —————————————————
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
            "options": ["Yes", "No"],
        },
        {
            "key": "interest",
            "system": "Choose one option. Deeply consider the choices and choose the one that best aligns with you as a person.",
            "user": "At what time would you drink soda?",
            "options": ["Morning", "Afternoon", "Evening"],
        },
        {
            "key": "timeline",
            "system": "Choose one option. Deeply consider the choices and choose the one that best aligns with you as a person.",
            "user": "How often do you drink soda?",
            "options": ["Frequently", "Occasionally", "Rarely", "Never"],
        },
    ]

# ——— 3) OpenAI helper ——————————————————————————
def call_chat(messages, fn=None, fn_name=None, temp=1.0):
    payload = {"model": "o4-mini", "messages": messages, "temperature": temp}
    if fn:
        payload["functions"] = [fn]
        payload["function_call"] = {"name": fn_name}
    return openai.chat.completions.create(**payload).choices[0].message

# ——— 4) Sidebar controls ————————————————————————
st.sidebar.header("Key Inputs")
industry = st.sidebar.text_input(
    "Industry/Product Name",
    value="Pepsi"
)
segment = st.sidebar.text_input(
    "Persona Segment (optional)",
    value="Health Buffs",
    help="Specify an optional subgroup label to guide persona generation (e.g., 'Health Buffs')."
)
n_personas = st.sidebar.number_input(
    "Number of personas",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
    help="How many synthetic personas to generate for the survey (between 5 and 50)."
)
run_button = st.sidebar.button("Run survey")
if run_button:
    # Visual feedback right under the button
    st.sidebar.info("🚀 Survey started — please wait on the results page!")

# ——— 5) Callbacks for add/remove rows ————————————
def remove_persona_field(idx):
    st.session_state.persona_fields.pop(idx)

def add_persona_field():
    intro_idx = next(
        (i for i,f in enumerate(st.session_state.persona_fields) if f["name"]=="intro"),
        len(st.session_state.persona_fields)
    )
    st.session_state.persona_fields.insert(intro_idx, {"name": "", "type": "string"})

def remove_question(idx):
    st.session_state.questions.pop(idx)

def add_question():
    st.session_state.questions.append({"key": "", "system": "", "user": "", "options": []})

# ——— 6) Tabs & editable UI ——————————————————————
tab_config, tab_results = st.tabs(["Configuration", "Results"])

with tab_config:
    st.header("Persona Attributes")
    for i, f in enumerate(st.session_state.persona_fields):
        if f["name"] == "intro":
            st.markdown("**intro** (fixed field)")
            continue
        c1, c2, c3 = st.columns([4, 2, 1])
        name = c1.text_input("Field name", f["name"], key=f"pf_name_{i}")
        typ  = c2.selectbox(
            "Type", ["string", "integer"],
            index=0 if f["type"]=="string" else 1,
            key=f"pf_type_{i}"
        )
        st.session_state.persona_fields[i] = {"name": name.strip(), "type": typ}
        c3.button("Remove", key=f"rm_pf_{i}", on_click=remove_persona_field, args=(i,))
    st.button("Add persona field", on_click=add_persona_field)

    st.markdown("---")
    st.header("Survey Questions")
    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Question {i+1}**")
        key  = st.text_input("Key", q["key"], key=f"q_key_{i}")
        sys  = st.text_input("System prompt", q["system"], key=f"q_sys_{i}")
        user = st.text_input("User prompt", q["user"], key=f"q_user_{i}")
        opts = st.text_input(
            "Options (comma-separated)",
            ", ".join(q["options"]),
            key=f"q_opts_{i}"
        )
        st.session_state.questions[i] = {
            "key": key.strip(),
            "system": sys.strip(),
            "user": user.strip(),
            "options": [o.strip() for o in opts.split(",") if o.strip()],
        }
        st.button("Remove question", key=f"rm_q_{i}", on_click=remove_question, args=(i,))
    st.button("Add survey question", on_click=add_question)

# ——— 7) Persona generation schema & fn —————————————
def build_persona_schema(fields):
    props, req = {}, []
    for f in fields:
        props[f["name"]] = {"type": "integer"} if f["type"]=="integer" else {"type": "string"}
        req.append(f["name"])
    return {"type":"object","properties":props,"required":req}

def make_persona_fn(schema):
    return {
        "name": "generate_personas",
        "description": "Generate customer personas.",
        "parameters": {
            "type":"object",
            "properties":{"personas":{"type":"array","items":schema}},
            "required":["personas"]
        }
    }

def generate_personas(segment, n, schema):
    fn      = make_persona_fn(schema)

    if segment:
        seg_txt = (
            f"All personas should share the background of “{segment},” "
            "but do not mention that label in the intros."
        )
    else:
        seg_txt = ""

    sys_msg = {
        "role": "system",
        "content": (
            f"Generate up to {n} unique customer personas. {seg_txt} "
            "Each persona should reflect that background with varied preferences, "
            "avoiding stereotypes. For each one, write a first person 4–5 sentence mini-biography "
            "that illustrates their background, motivations, and personal context—"
            "briefly mention the connection to the segment."
        )
    }

    personas, seen = [], set()
    while len(personas) < n:
        need = n - len(personas)
        resp = call_chat(
            [sys_msg, {"role":"user","content":f"Generate {need} personas now."}],
            fn=fn, fn_name="generate_personas"
        )
        batch = json.loads(resp.function_call.arguments)["personas"]
        for p in batch:
            uid = p.get("intro","") + p.get("name","")
            if uid not in seen:
                personas.append(p)
                seen.add(uid)
            if len(personas) >= n:
                break
    return personas

# ——— 8) Batch function schema —————————————————————
def make_batch_fn(questions):
    return {
        "name": "answer_survey",
        "description": "Answer multiple survey questions at once.",
        "parameters": {
            "type":"object",
            "properties": {
                **{q["key"]: {"type":"string", "enum": q["options"]} for q in questions}
            },
            "required": [q["key"] for q in questions]
        }
    }

# ——— 9) Parallelized batched survey runner —————————
def run_survey(personas, questions, segment):
    fields = list(st.session_state.persona_fields)
    qlist  = list(questions)

    scores   = {q["key"]: [] for q in qlist}
    batch_fn = make_batch_fn(qlist)

    def ask_persona(p):
        lines = [f"{f['name'].capitalize()}: {p.get(f['name'], '')}" for f in fields]
        if segment:
            lines.append(f"Segment: {segment}")
        base = {"role":"system", "content":"You are this persona:\n" + "\n".join(lines)}

        messages = [base] + [{"role":"system","content":q["system"]} for q in qlist]
        user_text = "\n".join(f"{i+1}. {q['user']}" for i, q in enumerate(qlist))
        messages.append({"role":"user", "content": user_text})

        resp    = call_chat(messages, fn=batch_fn, fn_name="answer_survey")
        return json.loads(resp.function_call.arguments)

    with ThreadPoolExecutor(max_workers=min(10, len(personas))) as exe:
        futures = {exe.submit(ask_persona, p): idx for idx, p in enumerate(personas)}
        for fut in as_completed(futures):
            answers = fut.result()
            for q in qlist:
                scores[q["key"]].append(answers.get(q["key"]))

    return scores

# ——— 10) Results tab ————————————————————————————
with tab_results:
    if run_button:
        schema     = build_persona_schema(st.session_state.persona_fields)
        questions  = st.session_state.questions

        with st.spinner("Generating personas…"):
            personas = generate_personas(segment, n_personas, schema)
        with st.spinner("Running survey in parallel…"):
            scores = run_survey(personas, questions, segment)

        st.title("Synthetic Survey Results")
        total = len(personas)

        # Per-question charts & tables with extra spacing
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
            st.write("")  # extra spacing

        # Personas & intros
        st.header("Generated Personas")
        st.dataframe(pd.DataFrame(personas))
        st.header("Persona Intros")
        for p in personas:
            st.markdown(f"**{p.get('name','')}**: {p.get('intro','')}")

        # Key Findings via LLM
        stats = {}
        for q in questions:
            opts = q["options"]
            dist = Counter(scores[q["key"]])
            stats[q["key"]] = {
                "counts":      {o: dist.get(o,0) for o in opts},
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
        find_resp = call_chat(
            find_prompt + [{"role":"system","content":"You are a data analyst."}],
            fn=find_fn, fn_name="generate_findings", temp=1
        )
        summary = json.loads(find_resp.function_call.arguments)["summary"]

        st.header("Key Findings")
        for para in summary.split("\n\n"):
            st.write(para)
