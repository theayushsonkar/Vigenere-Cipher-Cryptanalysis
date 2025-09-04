import streamlit as st
import vigenere_final  # import your script

st.title("🔑 Vigenère Cipher Cryptanalysis")
st.write("Enter your ciphertext (at least 150 words) for analysis:")

# Add external PDF link
st.markdown(
    '[📘 Read the Implementation Approach](https://drive.google.com/file/d/18UYd2UuCJJzBnNh8M5GL_hygux_dYNiM/view?usp=sharing)',
    unsafe_allow_html=True
)

ciphertext_input = st.text_area("Ciphertext:", height=300)

if st.button("Decrypt"):
    if len(ciphertext_input.split()) < 150:
        st.error("❌ Please enter at least 150 words of ciphertext.")
    else:
        result = vigenere_final.run_from_text(ciphertext_input)
        
        st.subheader("🔑 Key Found:")
        st.code(result["key"])
        
        st.subheader("📜 Plaintext:")
        st.text_area("Decrypted Text:", result["plaintext"], height=300)
