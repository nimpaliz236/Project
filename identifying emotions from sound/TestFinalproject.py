# gui_predict.py
import os
import wave
import numpy as np
import joblib
import simpleaudio as sa

from scipy import signal
from scipy.fftpack import dct

import torch
import torch.nn as nn

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# تنظیمات مسیر فایل‌های ذخیره‌شده1
MODEL_PATH = "best_model.pth"
SCALER_PATH = "scaler.pkl"
LE_PATH = "le.pkl"

TARGET_SR = 16000
TARGET_SECONDS = 3
TARGET_LENGTH = TARGET_SR * TARGET_SECONDS

# --- تعریف مدل (باید مطابق با مدل شما باشه) ---
class BetterMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.35),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# -----------------------------
# 2توابع کمکی (load_wav + MFCC aggregation) - عین کد اصلی، بدون librosa
def load_wav(path, target_sr=TARGET_SR, target_length=TARGET_LENGTH):
    with wave.open(path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            raise ValueError("Unsupported sample width: {}".format(sampwidth))

        audio = np.frombuffer(raw_data, dtype=dtype)

        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)
            audio = audio.mean(axis=1).astype(dtype)

        audio = audio.astype(np.float32) / (2**(8*sampwidth - 1))

        if framerate != target_sr:
            num_samples = int(len(audio) * float(target_sr) / framerate)
            audio = signal.resample(audio, num_samples)

        if len(audio) > target_length:
            start = (len(audio) - target_length) // 2
            audio = audio[start:start+target_length]
        elif len(audio) < target_length:
            pad = target_length - len(audio)
            audio = np.pad(audio, (0, pad), mode='constant')

        return audio, target_sr

# copy compute_mfcc_features from your script (simplified: returns aggregated vector)
# Note: change N_MFCC if your saved scaler expects different dim
N_MFCC = 20
N_FFT = 512
WIN_LEN = 0.025
WIN_STEP = 0.01
N_FILTERS = 40

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)
def mel_to_hz(mel):
    return 700.0 * (10**(mel / 2595.0) - 1.0)

def get_mel_filterbanks(n_filters=40, n_fft=512, sr=16000, low_freq=0, high_freq=None):
    if high_freq is None:
        high_freq = sr / 2
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fbanks = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        if center > left:
            fbanks[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fbanks[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fbanks

def delta(feat, N=2):
    denom = 2 * sum([i*i for i in range(1, N+1)])
    padded = np.pad(feat, ((N, N), (0,0)), mode='edge')
    delta_feat = np.zeros_like(feat)
    for t in range(feat.shape[0]):
        num = np.zeros((feat.shape[1],))
        for n in range(1, N+1):
            num += n * (padded[t + N + n] - padded[t + N - n])
        delta_feat[t] = num / denom
    return delta_feat

def compute_mfcc_features(signal_audio, sr=TARGET_SR,
                          n_mfcc=N_MFCC, n_fft=N_FFT,
                          win_len=WIN_LEN, win_step=WIN_STEP,
                          n_filters=N_FILTERS):
    pre_emphasis = 0.97
    emphasized = np.append(signal_audio[0], signal_audio[1:] - pre_emphasis * signal_audio[:-1])

    frame_length = int(round(win_len * sr))
    frame_step = int(round(win_step * sr))
    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_length - signal_length,))
    pad_signal = np.append(emphasized, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames *= np.hamming(frame_length)

    mag_frames = np.absolute(np.fft.rfft(frames, n=n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

    fbanks = get_mel_filterbanks(n_filters=n_filters, n_fft=n_fft, sr=sr)
    feat = np.dot(pow_frames, fbanks.T)
    feat[feat == 0] = np.finfo(float).eps
    log_feat = np.log(feat)

    mfcc = dct(log_feat, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    cep_lifter = 22
    nframes, ncoeff = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lifter / 2.) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift

    mfcc_delta = delta(mfcc, N=2)
    mfcc_delta2 = delta(mfcc_delta, N=2)

    # aggregate stats (mean,std,25th,75th) for mfcc, delta, delta2
    stats = []
    for arr in (mfcc, mfcc_delta, mfcc_delta2):
        stats.append(np.mean(arr, axis=0))
        stats.append(np.std(arr, axis=0))
        stats.append(np.percentile(arr, 25, axis=0))
        stats.append(np.percentile(arr, 75, axis=0))
    feature_vector = np.concatenate(stats)
    return feature_vector

# -----------------------------
# load scaler and label encoder and model
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(LE_PATH):
    raise FileNotFoundError("Required files not found: best_model.pth, scaler.pkl, le.pkl . Save them in the current folder.")

scaler = joblib.load(SCALER_PATH)
le = joblib.load(LE_PATH)

# create model and load weights (input dim from scaler shape)
input_dim = scaler.transform(np.zeros((1, scaler.mean_.shape[0]))).shape[1]
output_dim = len(le.classes_)
model = BetterMLP(input_dim, output_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# GUI
class AudioTesterApp:
    def __init__(self, root):
        self.root = root
        root.title("Audio Emotion Tester")
        root.geometry("800x600")

        self.filepath = None
        self.audio = None
        self.sr = None

        # top frame: buttons
        frm = ttk.Frame(root)
        frm.pack(pady=8)

        btn_load = ttk.Button(frm, text="Load WAV", command=self.load_file)
        btn_load.grid(row=0, column=0, padx=6)

        btn_play = ttk.Button(frm, text="Play", command=self.play_audio)
        btn_play.grid(row=0, column=1, padx=6)

        btn_predict = ttk.Button(frm, text="Predict", command=self.predict)
        btn_predict.grid(row=0, column=2, padx=6)

        # result label
        self.result_var = tk.StringVar(value="Prediction: -")
        lbl = ttk.Label(root, textvariable=self.result_var, font=("Helvetica", 14))
        lbl.pack(pady=8)

        # prob listbox
        self.prob_text = tk.Text(root, height=6)
        self.prob_text.pack(fill="x", padx=8)

        # matplotlib canvas for waveform
        self.fig, (self.ax1, self.ax2) = plt.subplots(2,1, figsize=(6,4))
        plt.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_file(self):
        fp = filedialog.askopenfilename(filetypes=[("WAV files","*.wav")])
        if not fp:
            return
        self.filepath = fp
        audio, sr = load_wav(fp)
        self.audio = audio
        self.sr = sr
        self.plot_waveform()
        self.result_var.set(f"Loaded: {os.path.basename(fp)}")

    def play_audio(self):
        if self.audio is None:
            messagebox.showwarning("No audio", "First load a WAV file.")
            return
        # convert float32 [-1,1] to int16 for simpleaudio
        audio_int16 = (self.audio * 32767).astype(np.int16)
        play_obj = sa.play_buffer(audio_int16.tobytes(), 1, 2, self.sr)
        # non-blocking

    def predict(self):
        if self.audio is None:
            messagebox.showwarning("No audio", "First load a WAV file.")
            return
        feat = compute_mfcc_features(self.audio, sr=self.sr)
        feat_scaled = scaler.transform(feat.reshape(1,-1))
        with torch.no_grad():
            out = model(torch.tensor(feat_scaled, dtype=torch.float32))
            probs = torch.softmax(out, dim=1).numpy().squeeze()
            pred_idx = int(np.argmax(probs))
            pred_label = le.inverse_transform([pred_idx])[0]

        self.result_var.set(f"Prediction: {pred_label} (idx={pred_idx})")
        # show probs
        self.prob_text.delete("1.0", tk.END)
        for i, cls in enumerate(le.classes_):
            self.prob_text.insert(tk.END, f"{cls}: {probs[i]:.3f}\n")
        # show bar plot of probs
        self.ax2.clear()
        self.ax2.bar(le.classes_, probs)
        self.ax2.set_ylabel("Prob")
        self.ax2.set_ylim(0,1)
        self.canvas.draw()

    def plot_waveform(self):
        self.ax1.clear()
        t = np.linspace(0, len(self.audio)/self.sr, len(self.audio))
        self.ax1.plot(t, self.audio)
        self.ax1.set_title("Waveform")
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioTesterApp(root)
    root.mainloop()  #The End
