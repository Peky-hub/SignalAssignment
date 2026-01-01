from hilbert_transformer import HilbertTransformer, HilbertModulator
import numpy as np
import soundfile as sf
from librosa import load

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

if __name__ == "__main__":
    AWGN_Simulation()