@st.cache
def load_app_text():
    with open(os.path.join("src", "integration", "app_text.md"), "r") as f:
        app_text = f.read()
    return app_text

def display_descriptions():
    st.header("Agent and Environment Descriptions")
    app_text = load_app_text()
    st.markdown(app_text)