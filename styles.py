"""Small, key-based CSS touches. Colours, fonts and radii come from .streamlit/config.toml."""

CSS = """
<style>
/* hide the (empty) container that holds this stylesheet */
[data-testid="stSidebar"] [data-testid="stHtml"] { display: none; }

/* sidebar items read like a list, not like centred buttons */
section[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
    justify-content: flex-start;
    text-align: left;
}
/* active chat row */
[class*="st-key-chatrow_on_"] [data-testid="stBaseButton-tertiary"] {
    background: rgba(128, 128, 128, 0.18);
    font-weight: 600;
}
/* home screen sits a little above the middle, like Claude / ChatGPT */
.st-key-home { padding-top: 9vh; }
/* keep long chat titles on one line */
section[data-testid="stSidebar"] [data-testid="stBaseButton-tertiary"] p {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
</style>
"""
