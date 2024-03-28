import streamlit as st

### Avec ce code (principalement streamlit), on peut faire le déployement sur le web

def main():
    st.title('Application WIADA 2023')
    user = st.text_input("Entrez votre nom : ")
    if st.button("Dis bonjour"):
        if user:
            st.write(f'Bonjour {user}')
        else :
            st.write('Bonjour à tous')
            
if __name__ == "__main__":
    main()