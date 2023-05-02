import streamlit as st
import requests
import string
import spacy

# Initialisation des stopwords
nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

def Preprocess_Sentence(Sentence):
    # Enlever la ponctuation
    Sentence = "".join([i.lower() for i in Sentence if i not in string.punctuation])
    # Enlever les chiffres
    Sentence = ''.join(i for i in Sentence if not i.isdigit())
    # Tokenization : Transformer les phrases en liste de tokens (en liste de mots)
    Sentence = nlp(Sentence)
    # Enlever les stopwords
    Sentence = [i.text for i in Sentence if i.text not in stopwords]
    # Enlever les mots qui ne sont pas alphabétiques ou qui ne sont pas dans le dictionnaire
    Sentence = ' '.join(w for w in Sentence if w.isalpha() and w.lower() in nlp.vocab)
    # Lemmatisation
    Sentence = ' '.join([token.lemma_ for token in nlp(Sentence)])
    
    return Sentence 

# Définir le titre de la page
st.set_page_config(page_title="Analyseur de sentiments")
# Titre de l'application
st.title("Analyse de sentiment par IA")
# Ajouter une image de fond
st.image("fond.png")
# Afficher une présentation
st.write("Bienvenue dans notre analyseur de sentiments !\n"
         "Celui-ci fonctionne avec un modèle de machine learning et a été entraîné sur 1 600 000 tweets.")
# Ajouter un champ de saisie pour la phrase
phrase = st.text_input("Entrez une phrase :")

# Ajouter un bouton pour lancer l'analyse
if st.button("Analyser"):
    # Envoyer la phrase à l'API et récupérer la prédiction
    sequence = Preprocess_Sentence(phrase)
    data = {'sequence': sequence}
    #response = requests.post("https://jglsr.pythonanywhere.com/prediction", json=data)
    #response = requests.post("https://p7-api-backend.herokuapp.com/prediction", json=data)
    response = requests.post("https://p7-backend.herokuapp.com/prediction", json=data)
    # Traiter la réponse de l'API
    if response.status_code == 200:
        result = response.json()
        if float(result['prediction']) < 0.2:
            st.write("La phrase est positive")
        elif float(result['prediction']) < 0.4:
            st.write("La phrase semble être positive")
        elif float(result['prediction']) < 0.6:
            st.write("La phrase semble être neutre")
        elif float(result['prediction']) < 0.8:
            st.write("La phrase semble être négative")
        else:
            st.write("La phrase est négative")
    else:
        st.write("Une erreur s'est produite lors de la requête à l'API.")
