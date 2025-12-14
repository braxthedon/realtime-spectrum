import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import wave
import time
import sys

# ---------- CONFIG ----------
SAMPLE_RATE = 44100      # Samples per second (CD-quality audio)
CHUNK_SIZE = 2048        # Number of samples per frame (controls update speed)
NUM_AVG_FRAMES = 5       # How many FFT frames to average for smoothing
MAX_SPEC_FRAMES = 80     # How many past frames to show in the spectrogram

# How many seconds of audio to keep for saving with 'r'
RECORD_SECONDS = 3
HISTORY_SAMPLES = SAMPLE_RATE * RECORD_SECONDS

# Frequency range to focus on (for plots + analysis)
FREQ_MIN = 20
FREQ_MAX = 8000

# Filter cutoffs
LP_CUTOFF = 1000.0        # Low-pass cutoff in Hz
HP_CUTOFF = 500.0         # High-pass cutoff in Hz

# Band-pass range (e.g. "telephone band")
BP_LOW = 300.0            # Band-pass low cutoff in Hz
BP_HIGH = 3400.0          # Band-pass high cutoff in Hz

# Notch filter defaults (adjustable live with keys)
NOTCH_FREQ_DEFAULT = 60.0
NOTCH_WIDTH_DEFAULT = 3.0

# Notch adjustment steps
NOTCH_STEP_HZ = 10.0      # '[' and ']' move notch center by this many Hz
NOTCH_WIDTH_STEP_HZ = 1.0 # '-' and '=' adjust width by this many Hz
NOTCH_FREQ_MIN = 20.0
NOTCH_FREQ_MAX = 8000.0
NOTCH_WIDTH_MIN = 1.0
NOTCH_WIDTH_MAX = 200.0


def save_audio_wav(path, audio_float, sample_rate):
    """Save mono float32 audio (-1..1) as 16-bit WAV."""
    audio_clipped = np.clip(audio_float, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    # ---------- PREP FFT FREQUENCIES ----------
    freqs = np.fft.rfftfreq(CHUNK_SIZE, d=1.0 / SAMPLE_RATE)
    freq_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    spec_freqs = freqs[freq_mask]

    # ---------- SET UP PLOTS ----------
    plt.ion()
    fig, (ax_time, ax_freq, ax_spec) = plt.subplots(3, 1, figsize=(10, 8))

    # ----- Time domain plot -----
    t = np.arange(CHUNK_SIZE) / SAMPLE_RATE
    line_time, = ax_time.plot(t, np.zeros(CHUNK_SIZE))
    ax_time.set_title("Real-Time Audio - Time Domain")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_ylim(-1.0, 1.0)
    ax_time.grid(True)

    # ----- Frequency domain plot -----
    line_freq, = ax_freq.plot(freqs[1:], np.zeros_like(freqs[1:]))  # skip 0 Hz
    ax_freq.set_title("Real-Time Audio - Frequency Domain (Magnitude)")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Amplitude")
    ax_freq.set_xscale("log")
    ax_freq.set_xlim(FREQ_MIN, FREQ_MAX)
    ax_freq.grid(True)

    # ----- Spectrogram plot -----
    spec_data = np.zeros((len(spec_freqs), MAX_SPEC_FRAMES))
    spec_image = ax_spec.imshow(
        spec_data,
        aspect="auto",
        origin="lower",
        extent=[-MAX_SPEC_FRAMES, 0, spec_freqs[0], spec_freqs[-1]],
    )
    ax_spec.set_title("Spectrogram (History of Frequency Content)")
    ax_spec.set_xlabel("Time (frames ago)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_ylim(FREQ_MIN, FREQ_MAX)
    ax_spec.grid(False)
    cbar = fig.colorbar(spec_image, ax=ax_spec, label="Intensity (dB-ish)")

    # ----- HUD TEXT TO THE RIGHT OF THE COLORBAR -----
    cbar_pos = cbar.ax.get_position()  # (x0, y0, x1, y1)
    hud_x = cbar_pos.x1 + 0.005        # just to the right of the bar
    hud_y1 = cbar_pos.y1
    hud_y2 = cbar_pos.y1 - 0.05
    hud_y3 = cbar_pos.y1 - 0.10

    hud_dom = fig.text(hud_x, hud_y1, "Dominant freq: -- Hz", fontsize=10, ha="left", va="top")
    hud_filter = fig.text(hud_x, hud_y2, "Filter: NONE", fontsize=10, ha="left", va="top")
    hud_level = fig.text(hud_x, hud_y3, "Level: -- dBFS", fontsize=10, ha="left", va="top")

    fig.tight_layout()

    print("Real-time audio spectrum analyzer (callback-based).")
    print("Keys (click the plot window first so it receives input):")
    print("  l = low-pass (toggle)")
    print("  h = high-pass (toggle)")
    print("  b = band-pass (toggle)")
    print("  n = notch (toggle)")
    print("  [ / ] = move notch center left/right")
    print("  - / = = make notch narrower/wider")
    print("  r = save last few seconds of audio + snapshot PNG")
    print("Ctrl+C in the terminal to quit.\n")

    # ---------- STATE ----------
    fft_history = []   # for smoothing spectrum
    audio_history = np.zeros(HISTORY_SAMPLES, dtype=np.float32)
    last_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)

    state = {
        "filter_mode": "none",   # 'none', 'lp', 'hp', 'bp', 'notch'
        "save_requested": False,
        "notch_freq": NOTCH_FREQ_DEFAULT,
        "notch_width": NOTCH_WIDTH_DEFAULT,
    }

    def update_filter_text():
        mode = state["filter_mode"]
        if mode == "lp":
            hud_filter.set_text(f"Filter: LOW-PASS (≤ {LP_CUTOFF:.0f} Hz)")
        elif mode == "hp":
            hud_filter.set_text(f"Filter: HIGH-PASS (≥ {HP_CUTOFF:.0f} Hz)")
        elif mode == "bp":
            hud_filter.set_text(f"Filter: BAND-PASS ({BP_LOW:.0f}–{BP_HIGH:.0f} Hz)")
        elif mode == "notch":
            nf = state["notch_freq"]
            nw = state["notch_width"]
            hud_filter.set_text(f"Filter: NOTCH ({nf:.0f} ± {nw:.0f} Hz)")
        else:
            hud_filter.set_text("Filter: NONE")

    update_filter_text()

    # ----- Filter selection helper -----
    def set_filter_mode(new_mode, label):
        if state["filter_mode"] == new_mode:
            state["filter_mode"] = "none"
            print(f"[KEY] Filter OFF (was {label})")
        else:
            state["filter_mode"] = new_mode
            print(f"[KEY] Filter mode: {label}")
        update_filter_text()

    # ----- Keyboard handler -----
    def on_key(event):
        k = event.key

        if k == "l":
            set_filter_mode("lp", "LOW-PASS")

        elif k == "h":
            set_filter_mode("hp", "HIGH-PASS")

        elif k == "b":
            set_filter_mode("bp", "BAND-PASS")

        elif k == "n":
            set_filter_mode("notch", "NOTCH")

        elif k == "[":
            state["notch_freq"] = clamp(state["notch_freq"] - NOTCH_STEP_HZ, NOTCH_FREQ_MIN, NOTCH_FREQ_MAX)
            print(f"[KEY] Notch center: {state['notch_freq']:.0f} Hz")
            if state["filter_mode"] == "notch":
                update_filter_text()

        elif k == "]":
            state["notch_freq"] = clamp(state["notch_freq"] + NOTCH_STEP_HZ, NOTCH_FREQ_MIN, NOTCH_FREQ_MAX)
            print(f"[KEY] Notch center: {state['notch_freq']:.0f} Hz")
            if state["filter_mode"] == "notch":
                update_filter_text()

        elif k == "-":
            state["notch_width"] = clamp(state["notch_width"] - NOTCH_WIDTH_STEP_HZ, NOTCH_WIDTH_MIN, NOTCH_WIDTH_MAX)
            print(f"[KEY] Notch width: ±{state['notch_width']:.0f} Hz")
            if state["filter_mode"] == "notch":
                update_filter_text()

        elif k in ["=", "+"]:
            # '=' and '+' are often the same key; accept both
            state["notch_width"] = clamp(state["notch_width"] + NOTCH_WIDTH_STEP_HZ, NOTCH_WIDTH_MIN, NOTCH_WIDTH_MAX)
            print(f"[KEY] Notch width: ±{state['notch_width']:.0f} Hz")
            if state["filter_mode"] == "notch":
                update_filter_text()

        elif k == "r":
            state["save_requested"] = True
            print("[KEY] Save requested: will write WAV + PNG with latest data.")

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ----- AUDIO CALLBACK (runs in separate thread) -----
    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_history, last_chunk
        if status:
            print(status, file=sys.stderr)

        samples = indata[:, 0].copy()

        # Update last_chunk (for visualization)
        last_chunk = samples

        # Update rolling audio history
        if frames < HISTORY_SAMPLES:
            audio_history = np.roll(audio_history, -frames)
            audio_history[-frames:] = samples
        else:
            audio_history = samples[-HISTORY_SAMPLES:]

    # ----- OPEN CONTINUOUS INPUT STREAM -----
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        channels=1,
        dtype="float32",
        callback=audio_callback
    ):
        try:
            while True:
                # Copy current chunk to avoid race issues
                chunk = last_chunk.copy()

                # ----- Time-domain plot -----
                line_time.set_ydata(chunk)

                # ----- Loudness meter (RMS -> dBFS) -----
                rms = np.sqrt(np.mean(chunk ** 2)) if chunk.size > 0 else 0.0
                if rms > 1e-8:
                    level_db = 20 * np.log10(rms + 1e-12)
                    hud_level.set_text(f"Level: {level_db:5.1f} dBFS")
                else:
                    hud_level.set_text("Level: -inf dBFS")

                # ----- FFT -----
                fft_vals = np.fft.rfft(chunk)
                magnitude = np.abs(fft_vals) / CHUNK_SIZE

                # ----- Apply filters based on mode -----
                filtered = magnitude.copy()
                mode = state["filter_mode"]

                if mode == "lp":
                    filtered[freqs > LP_CUTOFF] = 0.0

                elif mode == "hp":
                    filtered[freqs < HP_CUTOFF] = 0.0

                elif mode == "bp":
                    mask_outside = (freqs < BP_LOW) | (freqs > BP_HIGH)
                    filtered[mask_outside] = 0.0

                elif mode == "notch":
                    nf = state["notch_freq"]
                    nw = state["notch_width"]
                    notch_mask = (freqs > (nf - nw)) & (freqs < (nf + nw))
                    filtered[notch_mask] = 0.0

                # ----- Smoothing -----
                fft_history.append(filtered)
                if len(fft_history) > NUM_AVG_FRAMES:
                    fft_history.pop(0)
                smoothed = np.mean(fft_history, axis=0)

                # ----- Dominant frequency -----
                band_mags = smoothed[freq_mask]
                if band_mags.size > 0 and np.max(band_mags) > 1e-6:
                    peak_idx = np.argmax(band_mags)
                    dom_freq = spec_freqs[peak_idx]
                    hud_dom.set_text(f"Dominant freq: {dom_freq:.0f} Hz")
                else:
                    hud_dom.set_text("Dominant freq: -- Hz")

                # ----- Update spectrum line -----
                line_freq.set_ydata(smoothed[1:])

                # ----- Update spectrogram -----
                spec_column = smoothed[freq_mask]
                spec_column_db = 20 * np.log10(spec_column + 1e-8)

                spec_data = np.roll(spec_data, -1, axis=1)
                spec_data[:, -1] = spec_column_db

                spec_image.set_data(spec_data)
                spec_image.set_clim(vmin=np.min(spec_data), vmax=np.max(spec_data))

                # ----- Handle save request -----
                if state["save_requested"]:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    wav_name = f"recording_{timestamp}.wav"
                    png_name = f"snapshot_{timestamp}.png"

                    save_audio_wav(wav_name, audio_history, SAMPLE_RATE)
                    fig.savefig(png_name, dpi=150)

                    print(f"[SAVE] Wrote WAV: {wav_name}")
                    print(f"[SAVE] Wrote PNG: {png_name}")
                    state["save_requested"] = False

                # ----- Redraw -----
                fig.canvas.draw()
                fig.canvas.flush_events()
                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\nStopping analyzer.")
        finally:
            plt.ioff()
            plt.show()


if __name__ == "__main__":
    main()
