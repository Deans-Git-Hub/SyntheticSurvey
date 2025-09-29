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

# —— Helper: 1-dp percentages that sum to exactly 100.0 ——
def pct_1dp_sum_100(counts):
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    raw = [c * 100.0 / total for c in counts]
    rounded = [int(x * 10) / 10 for x in raw]  # floor to 0.1
    shortfall = round(100.0 - sum(rounded), 1)
    if shortfall > 0:
        order = sorted(range(len(raw)), key=lambda i: raw[i] - rounded[i], reverse=True)
        for i in order[:int(shortfall * 10)]:
            rounded[i] = round(rounded[i] + 0.1, 1)
    return [round(x, 1) for x in rounded]

# —— Helper: coerce model output to a valid enum (case/whitespace tolerant) ——
def coerce_to_enum(value, options):
    if value is None:
        return None
    s = str(value).strip().lower()
    for o in options:
        if s == o.strip().lower():
            return o  # return canonical casing from options
    return None

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
            "key": "Pricing",
            "system": "Choose one option. Deeply consider the choices and choose the one that best aligns with you as a person.",
            "user": "What is your perception of shoe prices in the last 5 years?",
            "options": ["Better, Worse"],
        },
        {
            "key": "Price Sensitivty",
            "system": "Choose one option. Deeply consider the choices and choose the one that best aligns with you as a person.",
            "user": "Given a 25% to 30% price increase, would you buy cheaper brands?",
            "options": ["Extremely likely, Likely, Neutral, Unlikely, Extremely unlikely"],
        },
        {
            "key": "Price Sensitivity",
            "system": "Choose one option. Deeply consider the choices and pick the one that best aligns with you as a persona.",
            "user": "3. Given a 25% to 30% price increase, would you increase total spend to buy all items you want?",
            "options": ["Extremely likely, Likely, Neutral, Unlikely, Extremely unlikely"],
        },
        {
            "key": "Retailer Preference",
            "system": "Choose one option. Deeply consider the choices and choose the one that best aligns with you as a person.",
            "user": "What type of retailer do you shop at for back-to-school purchases?",
            "options": ["Mass market, Specialty, Branded store or website, Discount/off-price"]
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
    value="Retail"
)
segment = st.sidebar.text_input(
    "Persona Segment (optional)",
    value="Parents buying kids' shoes for back-to-school",
    help="Specify an optional subgroup label to guide persona generation (e.g., 'Health Buffs')."
)
n_personas = st.sidebar.number_input(
    "Number of respondents",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
    help="How many synthetic respondents to generate for the survey (between 5 and 50)."
)
run_button = st.sidebar.button("Run survey")
if run_button:
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
            "properties": {**{q["key"]: {"type":"string", "enum": q["options"]} for q in questions}},
            "required": [q["key"] for q in questions]
        }
    }

# ——— 9) Parallelized batched survey runner —————————
def run_survey(personas, questions, segment):
    fields = list(st.session_state.persona_fields)
    qlist  = list(questions)

    scores   = {q["key"]: [] for q in qlist}
    batch_fn = make_batch_fn(qlist)

    def ask_persona(p, max_retries=2):
        # Build persona context
        lines = [f"{f['name'].capitalize()}: {p.get(f['name'], '')}" for f in fields]
        if segment:
            lines.append(f"Segment: {segment}")
        base = {"role":"system", "content":"You are this persona:\n" + "\n".join(lines)}

        # Guardrails to maximise valid, on-enum answers
        guard = {
            "role": "system",
            "content": (
                "You must answer every question. "
                "Select exactly one option for each key, using the options exactly as written."
            )
        }

        # Compose the question list once
        q_systems = [{"role":"system","content":q["system"]} for q in qlist]
        user_text = "\n".join(f"{i+1}. {q['user']}" for i, q in enumerate(qlist))
        user_msg  = {"role":"user", "content": user_text}

        attempt = 0
        while True:
            attempt += 1
            messages = [base, guard] + q_systems + [user_msg]
            resp = call_chat(messages, fn=batch_fn, fn_name="answer_survey", temp=1)
            raw = json.loads(resp.function_call.arguments)

            # Coerce & validate
            fixed = {}
            all_valid = True
            for q in qlist:
                val = coerce_to_enum(raw.get(q["key"]), q["options"])
                if val is None:
                    all_valid = False
                    break
                fixed[q["key"]] = val

            if all_valid or attempt > max_retries:
                return fixed if all_valid else {
                    # final fallback: replace any remaining invalid with the first option
                    q["key"]: fixed.get(q["key"], q["options"][0]) for q in qlist
                }

        # (unreachable)
        # return fixed

    # Parallel execution
    with ThreadPoolExecutor(max_workers=min(10, len(personas))) as exe:
        futures = {exe.submit(ask_persona, p): idx for idx, p in enumerate(personas)}
        for fut in as_completed(futures):
            answers = fut.result()
            for q in qlist:
                scores[q["key"]].append(answers[q["key"]])  # guaranteed valid

    return scores

# ——— 10) Results tab ————————————————————————————
with tab_results:
    if run_button:
        schema     = build_persona_schema(st.session_state.persona_fields)
        questions  = st.session_state.questions

        with st.spinner("Generating personas…"):
            personas = generate_personas(segment, n_personas, schema)
        with st.spinner("Running survey in parallel…"):
            scores = run_survey(personas, questions, segment)

        st.title("Synthetic Survey Results")
        st.caption(f"Total respondents: {len(personas)}")

        # Per-question charts & tables (larger charts)
        for q in questions:
            dist   = Counter(scores[q["key"]])     # now only valid options exist
            opts   = q["options"]
            counts = [dist.get(o, 0) for o in opts]
            perc   = pct_1dp_sum_100(counts)

            df = pd.DataFrame({
                "Option":  opts,
                "Count":   counts,
                "Percent": perc,
            })

            st.header(q["user"])
            chart = (
                alt.Chart(df)
                   .mark_bar()
                   .encode(
                       x=alt.X("Option:N", sort=opts),
                       y="Count:Q",
                       tooltip=["Option","Count","Percent"]
                   )
                   .properties(height=360)
            )
            st.altair_chart(chart, use_container_width=True)
            st.table(df.set_index("Option"))
            st.write("")

        # Personas & intros
        st.header("Synthetic Respondents")
        st.dataframe(pd.DataFrame(personas))
        st.header("Respondent Intros")
        for p in personas:
            st.markdown(f"**{p.get('name','')}**: {p.get('intro','')}")

        # Key Findings via LLM (uses same counts/perc)
        stats = {}
        for q in questions:
            dist   = Counter(scores[q["key"]])
            opts   = q["options"]
            counts = [dist.get(o, 0) for o in opts]
            perc   = pct_1dp_sum_100(counts)
            stats[q["key"]] = {
                "counts":      dict(zip(opts, counts)),
                "percentages": dict(zip(opts, perc)),
                "total":       sum(counts)
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
