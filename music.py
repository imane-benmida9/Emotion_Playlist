import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 
import os
import streamlit.components.v1 as components
from pytube import Search

# Configuration de la page
st.set_page_config(
    page_title="Music Emotion Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Réduire les logs TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DB954;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FFFFFF;
        margin-bottom: 1rem;
    }
    .emotion-display {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
        border: 2px solid #1DB954;
    }
    .recommendation-section {
        background: #2d2d2d;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
    }
    .stTextInput>div>div>input {
        background-color: #2d2d2d;
        color: white;
        border: 1px solid #1DB954;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<div class="main-header">🎵 Music Emotion Recommender</div>', unsafe_allow_html=True)

# Sidebar pour les informations
with st.sidebar:
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. **Configurez** votre langue et artiste préféré
    2. **Autorisez** l'accès à la caméra
    3. **Attendez** que votre émotion soit détectée
    4. **Cliquez** sur le bouton de recommandation
    """)
    
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    Cette application utilise l'IA pour:
    - Détecter vos émotions via la caméra
    - Recommander de la musique adaptée
    - Créer une playlist personnalisée
    """)

# Initialisation des variables de session
if 'emotion_detected' not in st.session_state:
    st.session_state.emotion_detected = ""
if 'recommendations_ready' not in st.session_state:
    st.session_state.recommendations_ready = False

# Section de configuration
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="sub-header">🎯 Préférences musicales</div>', unsafe_allow_html=True)
    lang = st.text_input("Langue (ex: French, English, Spanish)", placeholder="English")
    singer = st.text_input("Artiste préféré", placeholder="Ed Sheeran, Ariana Grande...")

# Chargement du modèle
@st.cache_resource
def load_emotion_model():
    try:
        model = load_model("model.h5", compile=False)
        label = np.load("labels.npy")
        return model, label
    except Exception as e:
        st.error(f"Erreur de chargement du modèle: {e}")
        return None, None

model, label = load_emotion_model()

# Initialisation MediaPipe
@st.cache_resource
def load_mediapipe():
    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    holis = holistic.Holistic()
    return holis, drawing, holistic, hands

holis, drawing, holistic, hands = load_mediapipe()

class EmotionProcessor:
    def recv(self, frame):
        frm = frame.to_ndarray(format="bgr24")
        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        lst = []
        if res.face_landmarks:
            # Traitement des landmarks du visage
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            # Traitement main gauche
            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            # Traitement main droite
            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if model is not None:
                lst = np.array(lst).reshape(1, -1)
                pred = label[np.argmax(model.predict(lst))]
                
                # Mise à jour de l'émotion détectée
                st.session_state.emotion_detected = pred
                np.save("emotion.npy", np.array([pred]))
                
                # Affichage de l'émotion sur le flux vidéo
                cv2.putText(frm, f"Emotion: {pred}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Dessin des landmarks
        if res.face_landmarks:
            drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        if res.left_hand_landmarks:
            drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        if res.right_hand_landmarks:
            drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# Section caméra
if lang and singer:
    st.markdown('<div class="sub-header">📷 Détection d\'émotion en temps réel</div>', unsafe_allow_html=True)
    
    col_cam1, col_cam2 = st.columns([2, 1])
    
    with col_cam1:
        webrtc_streamer(
            key="emotion-key",
            desired_playing_state=True,
            video_processor_factory=EmotionProcessor,
            media_stream_constraints={"video": True, "audio": False}
        )
    
    with col_cam2:
        if st.session_state.emotion_detected:
            st.markdown('<div class="emotion-display">', unsafe_allow_html=True)
            st.markdown(f"### 🎭 Émotion détectée")
            st.markdown(f"# {st.session_state.emotion_detected}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("🎭 L'émotion apparaîtra ici une fois détectée")

# Section recommandations
st.markdown("---")
st.markdown('<div class="sub-header">🎧 Recommandations musicales</div>', unsafe_allow_html=True)

btn = st.button("🎵 Recommander des musiques", use_container_width=True)

if btn:
    if not lang or not singer:
        st.error("⚠️ Veuillez d'abord remplir la langue et l'artiste")
    else:
        with st.spinner("🔍 Recherche de musiques adaptées..."):
            emotion_text = st.session_state.emotion_detected
            
            if not emotion_text:
                try:
                    loaded = np.load("emotion.npy", allow_pickle=True)
                    emotion_text = str(loaded[0]) if isinstance(loaded, np.ndarray) else str(loaded)
                    st.session_state.emotion_detected = emotion_text
                except:
                    emotion_text = ""

            if not emotion_text:
                st.warning("⏳ Veuillez attendre que votre émotion soit détectée par la caméra")
            else:
                # Affichage des paramètres de recherche
                st.markdown(f"""
                <div style='background: #1a1a1a; padding: 15px; border-radius: 10px; margin: 15px 0;'>
                    <h4>🔍 Paramètres de recherche:</h4>
                    <p><strong>Langue:</strong> {lang} | <strong>Artiste:</strong> {singer} | <strong>Émotion:</strong> {emotion_text}</p>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    # Recherche YouTube
                    search_query = f"{lang} {emotion_text} {singer} song"
                    s = Search(search_query)
                    results = s.results[:5]
                    video_ids = [video.video_id for video in results]

                    if video_ids:
                        # Génération de la playlist
                        first_video = video_ids[0]
                        playlist_str = ",".join(video_ids[1:])

                        embed_url = f"https://www.youtube.com/embed/{first_video}?playlist={playlist_str}&autoplay=1&loop=1"

                        # Affichage de la playlist
                        st.markdown('<div class="recommendation-section">', unsafe_allow_html=True)
                        st.markdown("### 🎶 Votre playlist personnalisée")
                        
                        components.html(
                            f"""
                            <iframe width="100%" height="450"
                                src="{embed_url}"
                                frameborder="0"
                                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                                allowfullscreen>
                            </iframe>
                            """,
                            height=500,
                        )
                        
                        # Liste des vidéos recommandées
                        st.markdown("### 📋 Titres recommandés")
                        for i, video in enumerate(results, 1):
                            st.write(f"{i}. {video.title}")
                            
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                    else:
                        st.error("❌ Aucune vidéo trouvée pour cette recherche. Essayez avec d'autres paramètres.")
                        
                except Exception as e:
                    st.error(f"❌ Erreur lors de la recherche: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; margin-top: 50px;'>"
    "🎵 Détection d'émotion et recommandation musicale • Powered by AI"
    "</div>", 
    unsafe_allow_html=True
)