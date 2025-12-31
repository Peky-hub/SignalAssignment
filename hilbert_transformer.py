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
import librosa


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

def _demo_using_test_audio():
    base = Path(__file__).parent
    test_audio = base / "TestAudio.wav"
    out_dir = base / "outputs"
    if not test_audio.exists():
        print(f"Test audio not found: {test_audio}")
        return

    data, sr = librosa.load(str(test_audio), sr=None, mono=True)

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

if __name__ == "__main__":
    _demo_using_test_audio()
