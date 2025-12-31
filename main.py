# 新增代码：音频读取、时域/频域分析、瞬时振幅/频率计算与保存图表/CSV
import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import csv
from pathlib import Path

test_audio_path = Path(__file__).parent.joinpath("TestAudio.wav")

def ensure_mono(data):
    if data.ndim == 1:
        return data
    return data.mean(axis=1)

def to_float32(data):
    # 将整数 PCM 转为 -1..1 浮点。若已为浮点则直接返回
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    # 根据位深推断最大值
    dtype = data.dtype
    if dtype == np.int16:
        maxv = 2**15
    elif dtype == np.int32:
        maxv = 2**31
    elif dtype == np.uint8:
        # uint8 WAV is offset by 128
        return ((data.astype(np.float32) - 128) / 128.0).astype(np.float32)
    else:
        maxv = np.abs(data).max() or 1.0
    return (data.astype(np.float32) / maxv).astype(np.float32)

def save_plot(fig, path):
    Path.mkdir(path.parent, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def analyze_audio(wav_path: Path, out_dir: Path):
    sr, data = wavfile.read(str(wav_path))
    data = ensure_mono(data)
    data = to_float32(data)
    duration = data.shape[0] / sr
    times = np.linspace(0, duration, num=data.shape[0], endpoint=False)

    Path.mkdir(out_dir, exist_ok=True)

    # 时域图
    fig = plt.figure(figsize=(10, 3))
    plt.plot(times, data, linewidth=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Time-domain waveform")
    save_plot(fig, out_dir / "waveform.png")

    # FFT 频谱分析
    N = len(data)
    # 汉宁窗
    window = np.hanning(N)
    fft_vals = np.fft.rfft(data * window)
    fft_freqs = np.fft.rfftfreq(N, 1.0 / sr)
    fft_amp = np.abs(fft_vals) / (N/2)

    # 频谱图
    fig = plt.figure(figsize=(10,4))
    plt.semilogy(fft_freqs, fft_amp + 1e-12)
    plt.xlim(0, sr/2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (linear, log-scale)")
    plt.title("Amplitude spectrum (FFT)")
    save_plot(fig, out_dir / "spectrum.png")

    # 时频图（Spectrogram）
    f, t, Sxx = signal.spectrogram(data, fs=sr, window='hann', nperseg=1024, noverlap=512, scaling='spectrum')
    fig = plt.figure(figsize=(10,4))
    plt.pcolormesh(t, f, 10*np.log10(Sxx+1e-12), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title("Spectrogram (dB)")
    plt.ylim(0, sr/2)
    save_plot(fig, out_dir / "spectrogram.png")

    # Hilbert -> 瞬时振幅与瞬时频率
    analytic = hilbert(data)
    inst_amplitude = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))
    # 瞬时频率：相位一阶差分
    inst_freq = np.empty_like(inst_phase)
    # 使用中点差分（保持与时间轴对齐）： df/dt = (phase[i+1]-phase[i])/(2*pi*dt)
    dt = 1.0 / sr
    dp = np.diff(inst_phase)
    # 最后一个点填充为与前一个相同的频率
    inst_freq[:-1] = (dp / (2.0 * np.pi * dt))
    inst_freq[-1] = inst_freq[-2] if len(inst_freq) > 1 else 0.0

    # 绘制瞬时幅度与频率（子图）
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(211)
    ax1.plot(times, inst_amplitude, color='C1', linewidth=0.5)
    ax1.set_ylabel("Instantaneous amplitude")
    ax1.set_title("Instantaneous amplitude & frequency")

    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(times, inst_freq, color='C2', linewidth=0.5)
    ax2.set_ylabel("Instantaneous frequency (Hz)")
    ax2.set_xlabel("Time (s)")
    save_plot(fig, out_dir / "inst_amp_freq.png")

    # 导出瞬时幅度/频率为 CSV（时间, amplitude, frequency）
    csv_path = out_dir / "instant_amp_freq.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "inst_amplitude", "inst_frequency_hz"])
        for i in range(len(times)):
            writer.writerow([f"{times[i]:.8f}", f"{inst_amplitude[i]:.6e}", f"{inst_freq[i]:.6f}"])

    # 打印基础摘要
    print(f"File: {wav_path}")
    print(f"Sample rate: {sr} Hz, Duration: {duration:.3f} s, Samples: {N}")
    print(f"Outputs saved to: {out_dir.resolve()}")

if __name__ == "__main__":
    out_dir = Path(__file__).parent.joinpath("outputs")
    if not test_audio_path.exists():
        print(f"Test audio not found: {test_audio_path}")
    else:
        analyze_audio(test_audio_path, out_dir)
