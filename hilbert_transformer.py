"""
HilbertTransformer

Provides a small, easy-to-use wrapper for generating an analytic signal
from a real input using the Hilbert transform (FFT-based). It computes
the instantaneous envelope, instantaneous phase and instantaneous
frequency and includes convenience plotting / CSV export helpers.

Usage:
    from Hilbert import HilbertTransformer
    ht = HilbertTransformer()
    analytic = ht.transform(x)
    envelope, phase = ht.envelope_phase(analytic)

This module also contains a small demo that will run when executed as
a script: it looks for `TestAudio.wav` next to this file and writes
outputs to `outputs/`.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from librosa import load
from soundfile import write

class HilbertTransformer:
    def analytic_signal(self, x: np.ndarray, N=None, axis=-1) -> np.ndarray:
        """使用希尔伯特变换生成解析信号, 参考自 scipy.signal.hilbert 的Marple实现

        x: 离散实值信号
        returns: 解析信号z = x + j*x_hat
        """
        """返回原始信号 x 的解析信号 (Analytic Signal)"""
        
        x = np.asarray(x)

        if N is None:
            N = x.shape[axis]
        if N < 1:
            raise ValueError("N must be a positive integer")
        
        # 1. 计算 FFT (使用原始长度 n)
        Xf = np.fft.fft(x, N, axis=axis)
        
        # 2. 构造频域掩码 h
        h = np.zeros(N)
        if N % 2 == 0:
            # 偶数长度
            h[0] = 1          # 直流项
            h[N // 2] = 1     # 奈奎斯特项
            h[1 : N // 2] = 2 # 正频率部分
        else:
            # 奇数长度
            h[0] = 1
            h[1 : (N + 1) // 2] = 2
        
        if x.ndim > 1:
            # 多维数组时，扩展掩码维度
            shape = [1] * x.ndim
            shape[axis] = N
            h = h.reshape(shape)
        
        # 3. 应用掩码并逆变换
        z = np.fft.ifft(Xf * h, axis=axis)
        return z

    def hilbert_transform(self, x: np.ndarray) -> np.ndarray:
        """希尔伯特变换，返回 x 的希尔伯特变换部分"""
        z = self.analytic_signal(x)
        return z.imag

    @staticmethod
    def envelope_phase(analytic: np.ndarray):
        """返回解析信号的瞬时包络和瞬时相位

        returns: (envelope, phase)
        """
        envelope = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        return envelope, phase

    @staticmethod
    def instantaneous_frequency(phase: np.ndarray, sr: float) -> np.ndarray:
        """计算瞬时频率

        phase: 瞬时相位
        sr: 采样率
        returns: 瞬时频率 (Hz)
        """
        # 相位的一阶差分
        dphase = np.diff(phase, prepend=phase[0])
        inst_freq = (sr / (2.0 * np.pi)) * dphase
        return inst_freq
    
    @staticmethod
    def plot_envelope_phase(out_path: Path, times: np.ndarray, envelope: np.ndarray, phase: np.ndarray):
        # Ensure output directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if phase is None or envelope is None:
            raise ValueError("Envelope and phase must be provided for plotting.")

        times = np.asarray(times)
        n_samples = times.shape[0]

        def _prep(y: np.ndarray, name: str):
            y = np.asarray(y)
            # 1D -> single channel
            if y.ndim == 1:
                if y.shape[0] != n_samples:
                    raise ValueError(f"{name} length {y.shape[0]} does not match times length {n_samples}")
                return y.reshape(1, n_samples)
            # Last axis matches time dimension -> treat trailing axis as samples
            if y.shape[-1] == n_samples:
                return y.reshape(-1, n_samples)
            # First axis matches time dimension -> transpose
            if y.shape[0] == n_samples:
                return y.T.reshape(-1, n_samples)
            raise ValueError(f"Could not align {name} with times; expected one dimension of length {n_samples}")

        env = _prep(envelope, "envelope")
        ph = _prep(phase, "phase")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        n_ch = env.shape[0]
        for ch in range(n_ch):
            label = f"ch{ch}" if n_ch > 1 else None
            ax1.plot(times, env[ch], linewidth=0.8, label=label)
            ax2.plot(times, ph[ch], linewidth=0.8, label=label)

        if n_ch > 1:
            ax1.legend(loc="upper right", fontsize="small")
            ax2.legend(loc="upper right", fontsize="small")

        ax1.set_ylabel("Envelope")
        ax2.set_ylabel("Phase (rad)")
        ax2.set_xlabel("Time (s)")

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

class HilbertModulator:
    """SSB 调制器（USB/LSB）

    基于 Hilbert 变换实现 SSB：
      x_ssb(t) = Re{ x_a(t) * exp(+j*2*pi*fc*t) }  (USB)
      x_ssb(t) = Re{ x_a(t) * exp(-j*2*pi*fc*t) } (LSB)

    demodulate 使用相干解调：乘以 2*cos(2*pi*fc*t) 并低通滤波得到基带信号。
    """

    def __init__(self):
        self.transformer = HilbertTransformer()

    def modulate(self, signal: np.ndarray, fc: float, sr: float, sideband: str = "usb") -> np.ndarray:
        """对实值基带信号进行 SSB 调制

        signal: 实值信号，最后一维为时间
        fc: 载波频率 (Hz)
        sr: 采样率 (Hz)
        sideband: 'usb' 或 'lsb'
        returns: 实值调制信号（同形状）
        """
        x = np.asarray(signal)
        # analytic_signal expects samples on last axis
        x_a = self.transformer.analytic_signal(x)

        n = x.shape[-1]
        t = np.arange(n) / float(sr)
        carrier = np.exp(1j * 2.0 * np.pi * fc * t)
        if sideband.lower() == "usb":
            ssb = np.real(x_a * carrier)
        elif sideband.lower() == "lsb":
            ssb = np.real(x_a * np.conj(carrier))
        else:
            raise ValueError("sideband must be 'usb' or 'lsb'")
        return ssb

    def demodulate(self, modulated_signal: np.ndarray, fc: float, sr: float, bw: float = None) -> np.ndarray:
        """对 SSB 信号进行相干解调

        modulated_signal: 实值 SSB 信号（最后一维为时间）
        fc: 载波频率
        sr: 采样率
        bw: 可选的低通滤波器截止频率 (Hz)。如果为 None，将使用 fc 作为截止（假设基带带宽小于 fc）。
        returns: 基带信号
        """
        y = np.asarray(modulated_signal)
        n = y.shape[-1]
        t = np.arange(n) / float(sr)

        # Multiply by complex exponential to shift the carrier to baseband
        y_a = self.transformer.analytic_signal(y)
        down = y_a * np.exp(-1j * 2.0 * np.pi * fc * t)

        # lowpass via FFT along last axis (use full FFT since `down` is complex)
        if bw is None:
            bw = fc
        Y = np.fft.fft(down, axis=-1)
        freqs = np.fft.fftfreq(n, d=1.0 / sr)
        # create symmetric mask around DC
        mask = np.abs(freqs) <= bw
        Y_filtered = Y * mask
        baseband_a = np.fft.ifft(Y_filtered, axis=-1)

        return np.real(baseband_a)

    def AM_Modulate(self, signal: np.ndarray, fc: float, sr: float, modulation_index: float = 1.0) -> np.ndarray:
        """对实值基带信号进行 AM 调制

        signal: 实值信号，最后一维为时间
        fc: 载波频率 (Hz)
        sr: 采样率 (Hz)
        modulation_index: 调制指数
        returns: 实值调制信号（同形状）
        """
        x = np.asarray(signal)
        n = x.shape[-1]
        t = np.arange(n) / float(sr)
        carrier = np.cos(2.0 * np.pi * fc * t)
        am_signal = (1.0 + modulation_index * x) * carrier
        return am_signal
    
    def AM_Demodulate(self, modulated_signal: np.ndarray, fc: float, sr: float, bw: float = None) -> np.ndarray:
        """对 AM 信号进行包络检波

        modulated_signal: 实值 AM 信号（最后一维为时间）
        fc: 载波频率
        sr: 采样率
        bw: 可选的低通滤波器截止频率 (Hz)。如果为 None，将使用 fc 作为截止（假设基带带宽小于 fc）。
        returns: 基带信号
        """
        y = np.asarray(modulated_signal)
        n = y.shape[-1]
        t = np.arange(n) / float(sr)

        # Multiply by 2*cos(2*pi*fc*t) to shift to baseband
        down = y * 2.0 * np.cos(2.0 * np.pi * fc * t)

        # lowpass via FFT along last axis
        if bw is None:
            bw = fc
        Y = np.fft.fft(down, axis=-1)
        freqs = np.fft.fftfreq(n, d=1.0 / sr)
        mask = np.abs(freqs) <= bw
        Y_filtered = Y * mask
        baseband = np.fft.ifft(Y_filtered, axis=-1)

        return np.real(baseband)

    @staticmethod
    def evaluate_spectral_efficiency(original: np.ndarray,
                                    modulated: np.ndarray,
                                    sr: float,
                                    power_frac: float = 0.99,
                                    exclude_carrier: bool = True):
        """
        使用占用带宽 (Occupied Bandwidth, OBW) 评估频谱效率。
        
        Returns: 
            dict: {
                'orig_bw': 原始基带信号带宽 (Hz),
                'mod_bw': 调制信号占用带宽 (Hz),
                'ratio': 带宽扩张倍数 (mod_bw / orig_bw)
            }
        """
        def _get_psd(sig):
            sig = np.asarray(sig)
            n = sig.shape[-1]
            # 获取单边频谱
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            s = np.fft.rfft(sig, axis=-1)
            psd = np.abs(s) ** 2
            if psd.ndim > 1:
                psd = psd.mean(axis=tuple(range(psd.ndim - 1)))
            return freqs, psd

        def _calculate_obw(freqs, psd, fraction):
            """计算包含 fraction 能量的频率宽度"""
            cum_p = np.cumsum(psd)
            total_p = cum_p[-1]
            if total_p <= 0:
                return 0.0
            
            low_limit = total_p * (1 - fraction) / 2
            high_limit = total_p * (1 + fraction) / 2
            
            idx_low = np.searchsorted(cum_p, low_limit)
            idx_high = np.searchsorted(cum_p, high_limit)
            
            return float(freqs[idx_high] - freqs[idx_low])

        # 1. 计算原始信号 PSD 及带宽
        freqs_o, psd_o = _get_psd(original)
        orig_bw = _calculate_obw(freqs_o, psd_o, power_frac)

        # 2. 计算调制信号 PSD
        freqs_m, psd_m = _get_psd(modulated)

        # 3. 针对 AM 信号剔除载波干扰
        # 如果不剔除载波，AM 的 OBW 会因为载波能量过大而显得极窄
        if exclude_carrier:
            # 仅在已调信号中寻找并剔除最强点（载波）
            idx_carrier = np.argmax(psd_m)
            # 窗口宽度设置为总频点的 0.5%，防止切掉边带
            win = max(1, int(len(freqs_m) * 0.005)) 
            lo = max(0, idx_carrier - win)
            hi = min(len(psd_m) - 1, idx_carrier + win)
            psd_m[lo:hi+1] = 0.0

        mod_bw = _calculate_obw(freqs_m, psd_m, power_frac)

        # 4. 计算比率
        # 理想 AM ratio 接近 0.5，理想 SSB ratio 接近 1.0
        ratio = float(orig_bw / mod_bw) if mod_bw > 0 else float('inf')

        return {
            'orig_bw': round(orig_bw, 2),
            'mod_bw': round(mod_bw, 2),
            'ratio': round(ratio, 4)
        }


def _demo_hilbert_transform():
    base = Path(__file__).parent
    test_audio = base / "TestAudio.wav"
    out_dir = base / "outputs"
    if not test_audio.exists():
        print(f"Test audio not found: {test_audio}")
        return

    data, sr = load(str(test_audio), sr=None, mono=True)

    ht = HilbertTransformer()
    analytic = ht.analytic_signal(data)
    envelope, phase = ht.envelope_phase(analytic)

    # compute number of samples from the last axis (works for mono and multi-channel)
    n_samples = data.shape[-1]
    duration = n_samples / float(sr)
    times = np.linspace(0, duration, num=n_samples, endpoint=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "hilbert_envelope_phase.png"

    ht.plot_envelope_phase(plot_path, times, envelope, phase)

    print(f"Wrote: {plot_path}")

def _demo_hilbert_modulator():
    base = Path(__file__).parent
    test_audio = base / "TestAudio.wav"
    out_dir = base / "outputs"
    if not test_audio.exists():
        print(f"Test audio not found: {test_audio}")
        return

    data, sr = load(str(test_audio), sr=None, mono=True)

    fc = 4400.0  # 4.4 kHz carrier
    modulator = HilbertModulator()
    ssb_signal = modulator.modulate(data, fc, sr, sideband="usb")
    am_signal = modulator.AM_Modulate(data, fc, sr)
    demodulated = modulator.demodulate(ssb_signal, fc, sr)

    # Evaluate spectral efficiency
    hilbert_efficiency = HilbertModulator.evaluate_spectral_efficiency(data, ssb_signal, sr, exclude_carrier=False)
    am_efficiency_no_carrier = HilbertModulator.evaluate_spectral_efficiency(data, am_signal, sr, exclude_carrier=True)

    print(f"Hilbert Efficiency (power): {hilbert_efficiency}")
    print(f"AM Efficiency (carrier excluded, power): {am_efficiency_no_carrier}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ssb_path = out_dir / "ssb_modulated.wav"
    am_path = out_dir / "am_modulated.wav"
    demod_path = out_dir / "ssb_demodulated.wav"

    write(str(ssb_path), ssb_signal, sr)
    write(str(am_path), am_signal, sr)
    write(str(demod_path), demodulated, sr)

    print(f"Wrote SSB modulated signal to: {ssb_path}")
    print(f"Wrote AM modulated signal to: {am_path}")
    print(f"Wrote demodulated signal to: {demod_path}")

if __name__ == "__main__":
    # _demo_hilbert_transform()
    _demo_hilbert_modulator()