from hilbert_transformer import HilbertTransformer, HilbertModulator
import numpy as np
import soundfile as sf
from librosa import load
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.signal as spsig

def test_demod_quality(original: np.ndarray, demodulated: np.ndarray, delay_compensation: bool = True):
    """
    评估解调质量：计算解调信号与原始信号的相似度与剩余噪声
    """
    # 1. 归一化幅度（消除调制增益影响）
    orig = original - np.mean(original)
    orig /= np.std(orig)
    
    demod = demodulated - np.mean(demodulated)
    demod /= np.std(demod)
    
    # 2. 对齐信号（补偿解调带来的群时延）
    # if delay_compensation:
    #     correlation = np.correlate(demod, orig, mode='full')
    #     delay = np.argmax(correlation) - (len(orig) - 1)
    #     demod = np.roll(demod, -delay)

    # 3. 计算均方误差 (MSE)
    mse = np.mean((orig - demod)**2)
    
    # 4. 计算解调后信噪比 (Post-demodulation SNR)
    # 假设误差完全由噪声引起
    signal_power = np.mean(orig**2)
    noise_power = mse
    post_snr = 10 * np.log10(signal_power / noise_power)
        
    return {
        'post_snr_db': round(post_snr, 2),
        'mse': round(mse, 6)
    }

def add_awgn(signal, target_snr_db):
    sig_avg_watts = np.mean(signal**2)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_avg_watts), len(signal))
    return signal + noise

def AWGN_Simulation():
    """模拟Additive White Gaussian Noise加性噪声环境，对信号进行调制解调处理，并使用信噪比评估"""
    signal, sr = load("TestAudio.wav", mono=True)
    awgn_signal = add_awgn(signal, target_snr_db=10)

    result = test_demod_quality(signal, awgn_signal)

    print(f"SNR: {result['post_snr_db']:.2f} dB")
    sf.write("outputs//Demodulated_AWGN.wav", awgn_signal, sr)

def analyze_test_audio(audio_path: str = "TestAudio.wav"):
    """Load TestAudio.wav and produce analysis plots:
      - time-domain waveform
      - power spectral density (Welch)
      - instantaneous envelope and phase (Hilbert)
      - instantaneous frequency

    Plots are saved to `outputs/`.
    """
    signal, sr = load(audio_path, mono=True)
    out_dir = Path("inputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    sig = np.asarray(signal)
    n = sig.shape[0]
    times = np.arange(n) / float(sr)

    # Time-domain waveform
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, sig, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Time-domain waveform")
    fig.tight_layout()
    time_path = out_dir / "time_domain.png"
    fig.savefig(time_path, dpi=150)
    plt.close(fig)

    # Power spectral density (Welch)
    f, Pxx = spsig.welch(sig, sr, nperseg=2048)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(f, 10.0 * np.log10(Pxx + 1e-15))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (dB/Hz)")
    ax.set_title("Power Spectral Density (Welch)")
    fig.tight_layout()
    psd_path = out_dir / "psd_welch.png"
    fig.savefig(psd_path, dpi=150)
    plt.close(fig)

    # Instantaneous envelope and phase via Hilbert
    ht = HilbertTransformer()
    analytic = ht.analytic_signal(sig)
    envelope, phase = ht.envelope_phase(analytic)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax1.plot(times, envelope, linewidth=0.5)
    ax1.set_ylabel("Envelope")
    ax2.plot(times, phase, linewidth=0.5)
    ax2.set_ylabel("Phase (rad)")
    ax2.set_xlabel("Time (s)")
    fig.tight_layout()
    env_phase_path = out_dir / "envelope_phase.png"
    fig.savefig(env_phase_path, dpi=150)
    plt.close(fig)

    # Instantaneous frequency
    inst_freq = ht.instantaneous_frequency(phase, sr)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(times, inst_freq, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Instantaneous frequency (Hz)")
    ax.set_title("Instantaneous Frequency")
    fig.tight_layout()
    inst_path = out_dir / "instantaneous_frequency.png"
    fig.savefig(inst_path, dpi=150)
    plt.close(fig)

    print(f"Wrote: {time_path}")
    print(f"Wrote: {psd_path}")
    print(f"Wrote: {env_phase_path}")
    print(f"Wrote: {inst_path}")

if __name__ == "__main__":
    # AWGN_Simulation()
    analyze_test_audio()