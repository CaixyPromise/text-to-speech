import os
import warnings
import wave

import paddle
import pyaudio
import soundfile as sf

warnings.filterwarnings("ignore")
from paddlespeech.t2s.frontend.zh_frontend import Frontend


# 声学模型路径
am_model_path = 'models/fastspeech2/model'
# 模型发声字典
phones_dict_path = 'models/fastspeech2/phone_id_map.txt'

# 声码器模型路径
voc_model_path = 'models/wavegan/model'

# 要合成的文本
text = '我是夜雨飘零，我爱深度学习！'
# 保存输出音频的路径
output_path = 'output/1.wav'


# 获取文本前端
frontend = Frontend(g2p_model='g2pM', phone_vocab_path=phones_dict_path)

# 声学模型
am_inference = paddle.jit.load(am_model_path)
# 声码器
voc_inference = paddle.jit.load(voc_model_path)

# 文本转模型输入
input_ids = frontend.get_input_ids(text, merge_sentences=False)
phone_ids = input_ids['phone_ids']

# 模型输出的拼接结果
wav_all = None
# 开始合成
for i in range(len(phone_ids)):
    part_phone_ids = phone_ids[i]
    # 获取声学模型的输出
    mel = am_inference(part_phone_ids)
    # 获取声码器模型输出
    wav = voc_inference(mel)
    # 如果是第一次生成就不用拼接数据
    if wav_all is None:
        wav_all = wav
    else:
        wav_all = paddle.concat([wav_all, wav])

# Tensor转Numpy
wav = wav_all.numpy()
# 保存音频
os.makedirs(os.path.dirname(output_path), exist_ok=True)
sf.write(output_path, wav, samplerate=24000)
print(f'音频已保存在：{output_path}')

chunk = 1024
wf = wave.open(output_path, 'rb')
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)
# 读取音频
data = wf.readframes(chunk)

# 循环读取全部数据
while len(data) > 0:
    stream.write(data)
    data = wf.readframes(chunk)

stream.stop_stream()
stream.close()
p.terminate()
