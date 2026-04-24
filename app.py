import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import ollama

# ---------------------------
# 🎨 UI STYLE (UPGRADE 2)
# ---------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background: #1c1f26;
    margin-bottom: 15px;
    border: 1px solid #2e2e3a;
}
.metric {
    font-size: 22px;
    font-weight: bold;
    color: #00ffcc;
}
.small-text {
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# 🔥 HERO HEADER (UPGRADE 1)
# ---------------------------
st.markdown("""
<h1 style='text-align: center;'>🚀 AI Skill Assessment & Learning Agent</h1>
<p style='text-align: center; color: grey;'>
Smart hiring powered by AI • Skill Matching • Resume Intelligence
</p>
""", unsafe_allow_html=True)

# ---------------------------
# ⚡ MODEL CACHE
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model_embed = load_model()

# ---------------------------
# MODE
# ---------------------------
mode = st.radio("Select Mode", ["Candidate", "Recruiter"])

voice_input = st.text_input("🎤 Voice Input")
job_desc = st.text_area("🧾 Job Description", value=voice_input)

# ---------------------------
# SKILLS
# ---------------------------
ALL_SKILLS = ["python", "sql", "machine learning", "power bi", "tableau", "excel", "statistics"]

def extract_skills(text):
    return [s for s in ALL_SKILLS if s in text.lower()]

def smart_score(resume, jd):
    res = extract_skills(resume)
    jd_s = extract_skills(jd)
    matched = len(set(res) & set(jd_s))
    total = len(jd_s) if jd_s else 1
    return int((matched / total) * 100), res, jd_s

def similarity_score(r, j):
    r, j = r[:1500], j[:1500]
    emb1 = model_embed.encode(r, convert_to_tensor=True)
    emb2 = model_embed.encode(j, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0][0]) * 100

# =========================================================
# 👤 CANDIDATE MODE (FINAL WITH PERSISTENCE)
# =========================================================
if mode == "Candidate":

    st.markdown("---")

    # ✅ SESSION STATE INIT
    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    if "res_skills" not in st.session_state:
        st.session_state.res_skills = []

    if "missing" not in st.session_state:
        st.session_state.missing = []

    if "final_score" not in st.session_state:
        st.session_state.final_score = 0

    if "suggestions" not in st.session_state:
        st.session_state.suggestions = ""

    # 📄 Upload
    file = st.file_uploader("📄 Upload Resume", type="pdf")

    text = ""
    if file:
        reader = PdfReader(file)
        for p in reader.pages:
            text += p.extract_text() or ""
        st.success("✅ Resume uploaded")

    # 🚀 ANALYZE BUTTON
    if st.button("🚀 Analyze with AI"):

        if not text or not job_desc:
            st.warning("⚠️ Upload resume and paste job description")
        else:
            with st.spinner("🤖 AI is analyzing..."):

                score, res_skills, jd_skills = smart_score(text, job_desc)
                sim = similarity_score(text, job_desc)
                final = int((score * 0.6) + (sim * 0.4))

                missing = list(set(jd_skills) - set(res_skills))

                # ✅ SAVE TO SESSION (IMPORTANT)
                st.session_state.analysis_done = True
                st.session_state.res_skills = res_skills
                st.session_state.missing = missing
                st.session_state.final_score = final

    # =====================================================
    # 📊 SHOW RESULT (PERSIST AFTER CLICK)
    # =====================================================
    if st.session_state.analysis_done:

        res_skills = st.session_state.res_skills
        missing = st.session_state.missing
        final = st.session_state.final_score

        # 🔥 SCORE CARD
        st.markdown(f"""
        <div class="card">
            <div class="metric">📊 Final Score: {final}%</div>
            <div class="small-text">AI combined score (skills + semantic)</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(final / 100)

        st.markdown("### ✔ Matched Skills")
        st.markdown(" ".join([f"`{s.upper()}`" for s in res_skills]) if res_skills else "None")

        st.markdown("### ❌ Missing Skills")
        st.markdown(" ".join([f"`{s.upper()}`" for s in missing]) if missing else "None")

        st.markdown("---")

        # =====================================================
        # 🤖 AI RESUME IMPROVEMENT
        # =====================================================
        with st.expander("💡 Improve Resume with AI"):

            if st.button("✨ Generate Suggestions", key="gen_suggestions"):

                try:
                    with st.spinner("🤖 Generating suggestions..."):

                        response = ollama.chat(
                            model="mistral",
                            messages=[{
                                "role": "user",
                                "content": f"""
You are a smart career coach.

Candidate Skills: {res_skills}
Missing Skills: {missing}

Give concise, practical, real-world suggestions:

1. Resume improvements (specific changes)
2. 2 strong project ideas (with tools & tech stack)
3. Exact skills to learn next (prioritized)
4. 1 hack to improve hiring chances

Keep it short, sharp, and impactful.
Avoid generic advice.
"""
                            }]
                        )

                        # ✅ SAVE
                        st.session_state.suggestions = response["message"]["content"]

                except:
                    st.error("⚠️ Ollama not running (run: ollama serve)")

            # ✅ SHOW ALWAYS
            if st.session_state.suggestions:
                st.markdown("### 🤖 Suggestions")
                st.write(st.session_state.suggestions)
# =========================================================
# 🧑‍💼 RECRUITER MODE (FINAL CLEAN ✅)
# =========================================================
elif mode == "Recruiter":

    st.markdown("""
    ## 🧑‍💼 Recruiter Dashboard  
    <span style='color: grey;'>Upload resumes → Analyze → Get insights instantly</span>
    """, unsafe_allow_html=True)

    files = st.file_uploader("📂 Upload Resumes", type="pdf", accept_multiple_files=True)

    # SESSION STORAGE
    if "df" not in st.session_state:
        st.session_state.df = None
        st.session_state.texts = {}

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # =============================
    # 🚀 ANALYZE
    # =============================
    if st.button("🚀 Analyze Candidates"):

        if not files or not job_desc:
            st.warning("⚠️ Upload resumes and add job description")
        else:
            data = {}
            results = []

            for f in files:
                reader = PdfReader(f)
                t = ""
                for p in reader.pages:
                    t += p.extract_text() or ""

                score, skills, _ = smart_score(t, job_desc)
                sim = similarity_score(t, job_desc)
                final = int((score * 0.6) + (sim * 0.4))

                data[f.name] = (t, skills)

                results.append({
                    "Candidate": f.name,
                    "Score": final
                })

            df = pd.DataFrame(results).sort_values(by="Score", ascending=False)

            st.session_state.df = df
            st.session_state.texts = data

    # =============================
    # 📊 SHOW RESULTS
    # =============================
    if st.session_state.df is not None:

        df = st.session_state.df
        texts = st.session_state.texts

        st.dataframe(df)

        top = df.iloc[0]["Candidate"]

        st.markdown(f"""
        <div class="card">
        🏆 <b>Top Candidate:</b> {top}
        </div>
        """, unsafe_allow_html=True)

        st.bar_chart(df.set_index("Candidate"))

        st.markdown("---")

        # =============================
        # 📄 DOWNLOAD REPORT
        # =============================
        def gen_pdf(df):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()
            content = []

            for _, r in df.iterrows():
                content.append(Paragraph(f"{r['Candidate']} - {r['Score']}%", styles["Normal"]))
                content.append(Spacer(1, 10))

            doc.build(content)
            buffer.seek(0)
            return buffer

        pdf = gen_pdf(df)

        st.download_button("📄 Download Report", pdf, "report.pdf")

    st.markdown("---")

    # =====================================================
    # 🤖 RECRUITER CHATBOT
    # =====================================================
    st.markdown("## 🤖 Recruiter Chatbot")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask about candidates")
        submit = st.form_submit_button("Ask AI")

    if submit and user_input:

        if st.session_state.df is None:
            st.warning("⚠️ Please analyze candidates first")
        else:
            try:
                df = st.session_state.df

                response = ollama.chat(
                    model="mistral",
                    messages=[{
                        "role": "user",
                        "content": f"""
You are a recruiter.

Candidates Data:
{df.to_string()}

Answer clearly and professionally.

Question:
{user_input}
"""
                    }]
                )

                answer = response["message"]["content"]

                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("AI", answer))

            except:
                st.error("⚠️ Run: ollama serve")

    # SHOW CHAT
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"🧑 **You:** {msg}")
        else:
            st.markdown(f"🤖 **AI:** {msg}")

    st.markdown("---")

    # =====================================================
    # 🧾 JOB DESCRIPTION GENERATOR
    # =====================================================
    st.markdown("## 🧾 Generate Job Description")

    role_input = st.text_input("Enter Role", key="jd_role")

    if st.button("✨ Generate JD", key="jd_button"):

        if not role_input:
            st.warning("⚠️ Enter role")
        else:
            try:
                response = ollama.chat(
                    model="mistral",
                    messages=[{
                        "role": "user",
                        "content": f"""
Generate a short job description for {role_input}

Use:
- Bullet points
- Clear sections
- Modern style
"""
                    }]
                )

                st.markdown(response["message"]["content"])

            except:
                st.error("⚠️ Run ollama serve")