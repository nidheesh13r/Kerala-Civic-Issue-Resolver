import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_lottie import st_lottie
import requests

# Set Streamlit page config
st.set_page_config(page_title="Kerala Civic Helpdesk", layout="wide")

# CSS Animations and Styling
st.markdown(
    """
<style>
body {
    background-color: white;
    color: black;
}
.fade-in {
    animation: fadeIn 1.5s ease forwards;
    opacity: 0;
    animation-fill-mode: forwards;
    animation-delay: 0.3s;
}
.slide-in {
    animation: slideIn 1s ease forwards;
    opacity: 0;
    animation-fill-mode: forwards;
    animation-delay: 0.5s;
}
@keyframes fadeIn {
    to { opacity: 1; }
}
@keyframes slideIn {
    from {
        transform: translateX(-30px);
        opacity: 0;
    }
    to {
        transform: translateX(0px);
        opacity: 1;
    }
}
.heading {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.subheading {
    font-size: 1.3rem;
    font-weight: 500;
    margin-bottom: 20px;
    color: #444;
}
.section-title {
    font-weight: 600;
    font-size: 1.2rem;
    margin-top: 25px;
    margin-bottom: 8px;
}
.solution-text {
    background-color: #f0f4f8;
    padding: 15px 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    animation: fadeIn 1.5s ease forwards;
}
ul {
    padding-left: 20px;
}
</style>
""",
    unsafe_allow_html=True,
)


# Load lottie animations helper
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Load your Lottie animations
lottie_title = load_lottieurl(
    "https://assets2.lottiefiles.com/packages/lf20_u4yrau.json"
)
lottie_input = load_lottieurl(
    "https://assets8.lottiefiles.com/packages/lf20_Cc8Bpg.json"
)
lottie_solution = load_lottieurl(
    "https://assets1.lottiefiles.com/packages/lf20_o1kaoo8l.json"
)


# Load data function
@st.cache_data
def load_data():
    problem_df = pickle.load(open("problem_df.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
    villages_df = pd.read_csv("Villages.csv")
    return problem_df, vectorizer, tfidf_matrix, villages_df


problem_df, vectorizer, tfidf_matrix, villages_df = load_data()

# Sidebar - Location selection
st.sidebar.title("📍 Select Your Location")
districts = sorted(villages_df["District"].unique())
selected_district = st.sidebar.selectbox("District", districts)
subdistricts = sorted(
    villages_df[villages_df["District"] == selected_district]["Sub-district"].unique()
)
selected_subdistrict = st.sidebar.selectbox("Sub-district", subdistricts)
villages = sorted(
    villages_df[
        (villages_df["District"] == selected_district)
        & (villages_df["Sub-district"] == selected_subdistrict)
    ]["Village"].unique()
)
selected_village = st.sidebar.selectbox("Village", villages)
st.sidebar.markdown(
    f"🧭 **Selected:** `{selected_district}` > `{selected_subdistrict}` > `{selected_village}`"
)

# Main UI layout
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown(
        '<div class="fade-in heading">🛡️ Kerala Civic Issue Resolver</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="fade-in subheading">Describe your civic issue to get an instant solution.</div>',
        unsafe_allow_html=True,
    )
with col2:
    if lottie_title:
        st_lottie(lottie_title, height=120, key="title_anim")

st.markdown("---")

col3, col4 = st.columns([3, 1])
with col3:
    user_problem = st.text_area(
        "📝 Type your problem below:",
        height=150,
        placeholder="Eg: Garbage not being collected in my area",
    )
    find_solution = st.button("🔍 Find Best Solution")
with col4:
    if lottie_input:
        st_lottie(lottie_input, height=150, key="input_anim")

if find_solution:
    if not user_problem.strip():
        st.warning("⚠️ Please enter a problem to proceed.")
    else:
        user_vec = vectorizer.transform([user_problem])
        similarity = cosine_similarity(user_vec, tfidf_matrix).flatten()
        top_idx = similarity.argmax()
        top_score = similarity[top_idx]

        if top_score < 0.1:
            st.info("❗ No close match found. Try describing your issue differently.")
        else:
            prob = problem_df.iloc[top_idx]

            st.markdown("---")
            st.markdown(
                '<div class="slide-in section-title">🛠️ Problem:</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="solution-text">{prob["Problem Description"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="slide-in section-title">🏢 Department:</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="solution-text">{prob["Department"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="slide-in section-title">✅ Solution Steps:</div>',
                unsafe_allow_html=True,
            )
            solution_steps = prob.get("Solution Steps", "")
            if isinstance(solution_steps, str) and "\n" in solution_steps:
                steps = solution_steps.strip().split("\n")
                st.markdown(
                    '<ul class="slide-in">'
                    + "".join(f"<li>{step.strip()}</li>" for step in steps)
                    + "</ul>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="solution-text">{solution_steps}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(f"**📊 Match Score:** {top_score:.2f}")

            if lottie_solution:
                st_lottie(lottie_solution, height=180, key="solution_anim")
