import streamlit as st
from rag_engine import compare_ingredients

st.title("🥦 NutriCheck AI")
st.subheader("Compare Two Products by Their Ingredients")

product_a = st.text_area("Ingredients of Product A", height=100)
product_b = st.text_area("Ingredients of Product B", height=100)
pref = st.text_input("Dietary Preference (optional)", placeholder="e.g., vegan, low-sugar")

if st.button("Compare Products"):
    if not product_a or not product_b:
        st.warning("Please enter ingredients for both products.")
    else:
        with st.spinner("Analyzing..."):
            result = compare_ingredients(product_a, product_b, dietary_pref=pref)
            st.success("Comparison Complete!")
            st.markdown(result)
