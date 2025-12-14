import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wave
import time
import sys

# ---------- CONFIG ----------
SAMPLE_RATE = 44100
CHUNK_SIZE = 2048
NUM_AVG_FRAMES = 5
MAX_SPEC_FRAMES = 80

RECORD_SECONDS = 3
HISTORY_SAMPLES = SAMPLE_RATE * RECORD_SECONDS

FREQ_MIN = 20
FREQ_MAX = 8000

LP_CUTOFF = 1000.0
HP_CUTOFF = 500.0

BP_LOW = 300.0
BP_HIGH = 3400.0

NOTCH_FREQ_DEFAULT = 60.0
NOTCH_WIDTH_DEFAULT = 3.0

NOTCH_STEP_HZ = 10.0
NOTCH_WIDTH_STEP_HZ = 1.0
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
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def main():
    freqs = np.fft.rfftfreq(CHUNK_SIZE, d=1.0 / SAMPLE_RATE)
    freq_mask = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    spec_freqs = freqs[freq_mask]

    # ---------- FIGURE LAYOUT (plots left, info panel right) ----------
    plt.ion()
    fig = plt.figure(figsize=(12, 7))  # slightly wider to fit the info panel

    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[4.2, 1.8],   # left plots vs right info panel
        height_ratios=[1, 1, 1],
        wspace=0.35,
        hspace=0.55
    )

    ax_time = fig.add_subplot(gs[0, 0])
    ax_freq = fig.add_subplot(gs[1, 0])
    ax_spec = fig.add_subplot(gs[2, 0])
    ax_info = fig.add_subplot(gs[:, 1])  # full height info panel
    ax_info.axis("off")

    # ----- Time plot -----
    t = np.arange(CHUNK_SIZE) / SAMPLE_RATE
    line_time, = ax_time.plot(t, np.zeros(CHUNK_SIZE))
    ax_time.set_title("Time Domain")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.set_ylim(-1.0, 1.0)
    ax_time.grid(True)

    # ----- Frequency plot -----
    line_freq, = ax_freq.plot(freqs[1:], np.zeros_like(freqs[1:]))  # skip 0 Hz
    ax_freq.set_title("Frequency Spectrum (FFT Magnitude)")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Amplitude")
    ax_freq.set_xscale("log")
    ax_freq.set_xlim(FREQ_MIN, FREQ_MAX)
    ax_freq.grid(True)

    # ----- Spectrogram -----
    spec_data = np.zeros((len(spec_freqs), MAX_SPEC_FRAMES))
    spec_image = ax_spec.imshow(
        spec_data,
        aspect="auto",
        origin="lower",
        extent=[-MAX_SPEC_FRAMES, 0, spec_freqs[0], spec_freqs[-1]],
    )
    ax_spec.set_title("Spectrogram")
    ax_spec.set_xlabel("Time (frames ago)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_ylim(FREQ_MIN, FREQ_MAX)

    # Keep colorbar but make it compact so it doesn't steal space
    cbar = fig.colorbar(spec_image, ax=ax_spec, fraction=0.046, pad=0.02)
    cbar.set_label("Intensity (dB-ish)")

    # ----- Info panel text object -----
    info_text = ax_info.text(
        0.0, 1.0, "",
        va="top",
        ha="left",
        fontsize=10,
        family="monospace"
    )

    # ---------- STATE ----------
    fft_history = []
    audio_history = np.zeros(HISTORY_SAMPLES, dtype=np.float32)
    last_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)

    state = {
        "filter_mode": "none",    # 'none', 'lp', 'hp', 'bp', 'notch'
        "save_requested": False,
        "notch_freq": NOTCH_FREQ_DEFAULT,
        "notch_width": NOTCH_WIDTH_DEFAULT,
        "dominant_freq": None,
        "level_dbfs": None,
    }

    def filter_label():
        mode = state["filter_mode"]
        if mode == "lp":
            return f"LOW-PASS (≤ {LP_CUTOFF:.0f} Hz)"
        if mode == "hp":
            return f"HIGH-PASS (≥ {HP_CUTOFF:.0f} Hz)"
        if mode == "bp":
            return f"BAND-PASS ({BP_LOW:.0f}–{BP_HIGH:.0f} Hz)"
        if mode == "notch":
            return f"NOTCH ({state['notch_freq']:.0f} ± {state['notch_width']:.0f} Hz)"
        return "NONE"

    def update_info_panel():
        dom = "--" if state["dominant_freq"] is None else f"{state['dominant_freq']:.0f} Hz"
        lvl = "--" if state["level_dbfs"] is None else f"{state['level_dbfs']:.1f} dBFS"

        text = (
            "=== CONTROLS ===\n"
            "l : low-pass\n"
            "h : high-pass\n"
            "b : band-pass\n"
            "n : notch\n"
            "[ ] : move notch\n"
            "- = : width\n"
            "r : save WAV+PNG\n\n"
            "=== STATUS ===\n"
            f"Filter : {filter_label()}\n"
            f"Dominant: {dom}\n"
            f"Level  : {lvl}\n\n"
            "=== NOTCH ===\n"
            f"Center : {state['notch_freq']:.0f} Hz\n"
            f"Width  : ±{state['notch_width']:.0f} Hz\n\n"
            "=== BAND-PASS ===\n"
            f"{BP_LOW:.0f}–{BP_HIGH:.0f} Hz\n"
        )
        info_text.set_text(text)

    update_info_panel()

    def set_filter_mode(new_mode, label):
        # Toggle: pressing same key again turns it off
        if state["filter_mode"] == new_mode:
            state["filter_mode"] = "none"
            print(f"[KEY] Filter OFF (was {label})")
        else:
            state["filter_mode"] = new_mode
            print(f"[KEY] Filter mode: {label}")
        update_info_panel()

    # ---------- KEYBOARD HANDLER ----------
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
            update_info_panel()
        elif k == "]":
            state["notch_freq"] = clamp(state["notch_freq"] + NOTCH_STEP_HZ, NOTCH_FREQ_MIN, NOTCH_FREQ_MAX)
            print(f"[KEY] Notch center: {state['notch_freq']:.0f} Hz")
            update_info_panel()
        elif k == "-":
            state["notch_width"] = clamp(state["notch_width"] - NOTCH_WIDTH_STEP_HZ, NOTCH_WIDTH_MIN, NOTCH_WIDTH_MAX)
            print(f"[KEY] Notch width: ±{state['notch_width']:.0f} Hz")
            update_info_panel()
        elif k in ["=", "+"]:
            state["notch_width"] = clamp(state["notch_width"] + NOTCH_WIDTH_STEP_HZ, NOTCH_WIDTH_MIN, NOTCH_WIDTH_MAX)
            print(f"[KEY] Notch width: ±{state['notch_width']:.0f} Hz")
            update_info_panel()
        elif k == "r":
            state["save_requested"] = True
            print("[KEY] Save requested: will write WAV + PNG with latest data.")

    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Running analyzer (click the plot window first so it receives key input).")
    print("Keys: l/h/b/n, [ ], -/=, r. Ctrl+C to quit.\n")

    # ---------- AUDIO CALLBACK ----------
    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_history, last_chunk
        if status:
            print(status, file=sys.stderr)

        samples = indata[:, 0].copy()
        last_chunk = samples

        # rolling buffer for last RECORD_SECONDS
        if frames < HISTORY_SAMPLES:
            audio_history = np.roll(audio_history, -frames)
            audio_history[-frames:] = samples
        else:
            audio_history = samples[-HISTORY_SAMPLES:]

    # ---------- STREAM + MAIN LOOP ----------
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=CHUNK_SIZE,
        channels=1,
        dtype="float32",
        callback=audio_callback
    ):
        try:
            while True:
                chunk = last_chunk.copy()

                # Time plot
                line_time.set_ydata(chunk)

                # Loudness (RMS -> dBFS)
                rms = np.sqrt(np.mean(chunk ** 2)) if chunk.size > 0 else 0.0
                if rms > 1e-8:
                    state["level_dbfs"] = 20 * np.log10(rms + 1e-12)
                else:
                    state["level_dbfs"] = None

                # FFT
                fft_vals = np.fft.rfft(chunk)
                magnitude = np.abs(fft_vals) / CHUNK_SIZE

                # Apply filter (frequency-domain masking)
                filtered = magnitude.copy()
                mode = state["filter_mode"]

                if mode == "lp":
                    filtered[freqs > LP_CUTOFF] = 0.0
                elif mode == "hp":
                    filtered[freqs < HP_CUTOFF] = 0.0
                elif mode == "bp":
                    outside = (freqs < BP_LOW) | (freqs > BP_HIGH)
                    filtered[outside] = 0.0
                elif mode == "notch":
                    nf = state["notch_freq"]
                    nw = state["notch_width"]
                    notch_mask = (freqs > (nf - nw)) & (freqs < (nf + nw))
                    filtered[notch_mask] = 0.0

                # Smoothing
                fft_history.append(filtered)
                if len(fft_history) > NUM_AVG_FRAMES:
                    fft_history.pop(0)
                smoothed = np.mean(fft_history, axis=0)

                # Dominant frequency
                band = smoothed[freq_mask]
                if band.size > 0 and np.max(band) > 1e-6:
                    peak_idx = int(np.argmax(band))
                    state["dominant_freq"] = spec_freqs[peak_idx]
                else:
                    state["dominant_freq"] = None

                # Update info panel (so it always shows current numbers)
                update_info_panel()

                # Spectrum plot
                line_freq.set_ydata(smoothed[1:])

                # Spectrogram update
                spec_column = smoothed[freq_mask]
                spec_column_db = 20 * np.log10(spec_column + 1e-8)

                spec_data[:] = np.roll(spec_data, -1, axis=1)
                spec_data[:, -1] = spec_column_db

                spec_image.set_data(spec_data)
                spec_image.set_clim(vmin=np.min(spec_data), vmax=np.max(spec_data))

                # Save request
                if state["save_requested"]:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    wav_name = f"recording_{timestamp}.wav"
                    png_name = f"snapshot_{timestamp}.png"

                    save_audio_wav(wav_name, audio_history, SAMPLE_RATE)
                    fig.savefig(png_name, dpi=150)

                    print(f"[SAVE] Wrote WAV: {wav_name}")
                    print(f"[SAVE] Wrote PNG: {png_name}")
                    state["save_requested"] = False

                # Redraw
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
