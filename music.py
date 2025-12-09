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

# CSS personnalisé avec fond violet
st.markdown("""
<style>
    /* Fond violet pour toute l'application */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        font-size: 3rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .landing-title {
        font-size: 4rem;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .landing-subtitle {
        font-size: 1.8rem;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        opacity: 0.9;
    }
    
    .info-section {
        background: rgba(255, 255, 255, 0.15);
        padding: 30px;
        border-radius: 20px;
        margin: 10px;
        text-align: left;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        height: 100%;
    }
    
    .info-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
    
    .info-content {
        font-size: 1.1rem;
        line-height: 1.8;
        color: rgba(255, 255, 255, 0.95);
    }
    
    .start-button-container {
        text-align: center;
        margin: 40px 0;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: white;
        margin-bottom: 1rem;
    }
    
    .emotion-display {
        background: rgba(255, 255, 255, 0.2);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 20px 0;
        border: 2px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .recommendation-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #ffffff 0%, #f0f0f0 100%);
        color: #667eea;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: bold;
        width: 100%;
    }
    
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.9);
        color: #333;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* Style pour les messages info */
    .stInfo {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
    }
    
    /* Style pour les messages d'erreur et warning */
    .stAlert {
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialisation des variables de session
if 'app_started' not in st.session_state:
    st.session_state.app_started = False
if 'emotion_detected' not in st.session_state:
    st.session_state.emotion_detected = ""
if 'recommendations_ready' not in st.session_state:
    st.session_state.recommendations_ready = False

# PAGE D'ACCUEIL
if not st.session_state.app_started:
    # Titre principal
    st.markdown('<div class="landing-title">MUSIC EMOTION</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-title">RECOMMENDER</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-subtitle">Discover Music That Feels Like You.</div>', unsafe_allow_html=True)
    
    # Sections en colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-section">
            <div class="info-title">📋 Instructions</div>
            <div class="info-content">
                <p><strong>Suivez ces étapes :</strong></p>
                <ol>
                    <li><strong>Configurez</strong> votre langue et artiste préféré</li>
                    <li><strong>Autorisez</strong> l'accès à la caméra</li>
                    <li><strong>Attendez</strong> que votre émotion soit détectée</li>
                    <li><strong>Cliquez</strong> sur le bouton de recommandation</li>
                </ol>
                <p><em>L'analyse est instantanée !</em></p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-section">
            <div class="info-title">ℹ À propos</div>
            <div class="info-content">
                <p><strong>Cette application utilise l'IA pour :</strong></p>
                <ul>
                    <li>Détecter vos émotions via la caméra</li>
                    <li>Recommander de la musique adaptée</li>
                    <li>Créer une playlist personnalisée</li>
                </ul>
                <br>
                <p style="text-align: center; border-top: 1px solid rgba(255,255,255,0.3); padding-top: 15px;">
                    <strong>🎵 Powered by AI</strong>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Bouton Commencer
    st.markdown('<div class="start-button-container">', unsafe_allow_html=True)
    if st.button("🚀 Commencer l'expérience", key="start_btn", use_container_width=True):
        st.session_state.app_started = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# ============================================================================
# APPLICATION PRINCIPALE
# ============================================================================

# Header principal
st.markdown('<div class="main-header">🎵 Music Emotion Recommender</div>', unsafe_allow_html=True)

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
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
        st.error("⚠ Veuillez d'abord remplir la langue et l'artiste")
    else:
        # Relire l'émotion la plus récente depuis le fichier
        try:
            loaded = np.load("emotion.npy", allow_pickle=True)
            st.session_state.emotion_detected = str(loaded[0])
        except:
            st.session_state.emotion_detected = ""

        emotion_text = st.session_state.emotion_detected

        if not emotion_text:
            st.warning("⏳ Veuillez attendre que votre émotion soit détectée par la caméra")
        else:
            with st.spinner("🔍 Recherche de musiques adaptées..."):
                # Affichage des paramètres de recherche
                st.markdown(f"""
                <div style='background: rgba(255, 255, 255, 0.15); padding: 15px; border-radius: 10px; margin: 15px 0;'>
                    <h4 style='color: white;'>🔍 Paramètres de recherche:</h4>
                    <p style='color: white;'><strong>Langue:</strong> {lang} | <strong>Artiste:</strong> {singer} | <strong>Émotion:</strong> {emotion_text}</p>
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
    "<div style='text-align: center; color: white; margin-top: 50px; opacity: 0.8;'>"
    "🎵 Détection d'émotion et recommandation musicale • Powered by AI"
    "</div>", 
    unsafe_allow_html=True
)