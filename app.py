import streamlit as st
import time
import numpy as np
from achumeter.audio_utils import record_audio, save_wav, extract_features, compute_rms_over_time, make_slowmo
from achumeter.model import AchuModel
import tempfile, os
import pyttsx3
from pydub import AudioSegment
from threading import Thread
import matplotlib.pyplot as plt
import simpleaudio as sa


st.set_page_config(page_title="Acheeww Meter — Incoming Sneeze", layout="centered")

st.title("🤧 Acheeww Meter — Predict How Close a Sneeze Is")
st.write("Record 3–5 seconds of pre-sneeze audio and get a dramatic prediction.")

RECORD_SECONDS = st.slider("Recording duration (seconds)", 2, 6, 4)
model = AchuModel()

def play_sound_async(path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(path)
        # Play in background thread so UI isn't blocked
        Thread(target=lambda: wave_obj.play()).start()
    except Exception as e:
        print("playback failed:", e)

col1, col2 = st.columns(2)
with col1:
    if st.button("Record from microphone"):
        st.info("Recording... speak/sniff into your mic")
        audio = record_audio(RECORD_SECONDS)
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        save_wav(tmp.name, audio, sr=22050)
        st.success(f"Saved {tmp.name}")
        st.audio(tmp.name)
        features = extract_features(tmp.name)
        rms_series = compute_rms_over_time(tmp.name, frame_length=1024, hop_length=512)
        time_pred, intensity = model.predict(features)
        st.markdown("---")
        st.warning(" Harii 🤧 ")
        st.header("Prediction")
        st.write(f"⏳ **Time until sneeze:** {time_pred:.2f} s")
        st.write(f"🌪️ **Intensity:** **{intensity.upper()}**")
        if intensity.lower() == 'high':
            st.warning("⚠️ High intensity detected — move the mic away!")
        # SneezeGraph
        fig, ax = plt.subplots(figsize=(6,2))
        t = np.linspace(0, len(rms_series) * (512/22050), num=len(rms_series))
        ax.plot(t, rms_series)
        ax.set_xlabel("Seconds")
        ax.set_ylabel("RMS (energy)")
        ax.set_title("SneezeGraph™ — Buildup vs Time")
        st.pyplot(fig)
        # Buttons
        if st.button("Start Sneeze Timer (countdown)"):
            placeholder = st.empty()
            total = max(0.1, float(time_pred))
            end_time = time.time() + total
            while time.time() < end_time:
                left = end_time - time.time()
                placeholder.markdown(f"### ⏱️ Incoming in **{left:.2f}s** — take cover!")
                time.sleep(0.05)
            placeholder.markdown("### 💥 BOOM! Sneeze detonated!")
            explosion_path = os.path.join(os.path.dirname(__file__), "achumeter", "sounds", "explosion.wav")
            play_sound_async(explosion_path)
            st.balloons()
            st.success("Sneeze detonated — hope your screen survived!")
        if st.button("Sneeze Shield Warning (Protect your screen)"):
            st.warning("🛡️ SneezeShield™: Cover your screen or step back!")
            st.info("Tip: Use a physical shield, press 's' to simulate, or click the Protect button.")
        if st.button("Sneeze Replay (Slow‑mo)"):
            slow_path = make_slowmo(tmp.name, speed_factor=0.5)
            st.audio(slow_path)
            music_path = os.path.join(os.path.dirname(__file__), "achumeter", "sounds", "cinematic.mp3")
            try:
                music = AudioSegment.from_file(music_path)
                sneeze = AudioSegment.from_file(slow_path)
                music = music - 12
                combined = sneeze.overlay(music[:len(sneeze)])
                combined_path = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
                combined.export(combined_path, format="mp3")
                st.audio(combined_path)
            except Exception as e:
                print("Could not overlay music:", e)
        if st.button("Announce prediction (TTS)"):
            engine = pyttsx3.init()
            msg = f'Incoming sneeze in {time_pred:.1f} seconds. Intensity {intensity}.'
            engine.say(msg)
            engine.runAndWait()

with col2:
    uploaded = st.file_uploader("Or upload a WAV/MP3 file", type=['wav','mp3'])
    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp.write(uploaded.getbuffer())
        tmp.flush()
        st.audio(tmp.name)
        features = extract_features(tmp.name)
        rms_series = compute_rms_over_time(tmp.name, frame_length=1024, hop_length=512)
        time_pred, intensity = model.predict(features)
        st.markdown("---")
        st.header("Prediction")
        st.write(f"⏳ **Time until sneeze:** {time_pred:.2f} s")
        st.write(f"🌪️ **Intensity:** **{intensity.upper()}**")
        fig, ax = plt.subplots(figsize=(6,2))
        t = np.linspace(0, len(rms_series) * (512/22050), num=len(rms_series))
        ax.plot(t, rms_series)
        ax.set_xlabel("Seconds")
        ax.set_ylabel("RMS (energy)")
        ax.set_title("SneezeGraph™ — Buildup vs Time")
        st.pyplot(fig)
        if st.button("Sneeze Replay (Slow‑mo) - uploaded"):
            slow_path = make_slowmo(tmp.name, speed_factor=0.5)
            st.audio(slow_path)
            music_path = os.path.join(os.path.dirname(__file__), "achumeter", "sounds", "cinematic.mp3")
            try:
                music = AudioSegment.from_file(music_path)
                sneeze = AudioSegment.from_file(slow_path)
                music = music - 12
                combined = sneeze.overlay(music[:len(sneeze)])
                combined_path = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False).name
                combined.export(combined_path, format="mp3")
                st.audio(combined_path)
            except Exception as e:
                print("Could not overlay music:", e)

st.markdown("---")
st.write("**Pro tip:** This is a fun demo. Improve accuracy by training with many labeled samples and switching to an LSTM/CNN model for temporal patterns.")
