import os
import time
import warnings

import paddle
import soundfile as sf
from flasgger import Swagger
from flask import Flask, request, send_file
from flask_cors import CORS

warnings.filterwarnings("ignore")
from paddlespeech.t2s.frontend.zh_frontend import Frontend

# 读取服务配置文件
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
# 允许跨越访问
CORS(app)
# 文档配置
app.config['SWAGGER'] = {
    'title': '夜雨飘零语音合成服务',
    'uiversion': 3
}
Swagger(app)

# ===========================快速模型===================================
# 声学模型路径
am_fastspeech2_model_path = 'models/fastspeech2/model'
# 模型发声字典
fastspeech2_phones_dict_path = 'models/fastspeech2/phone_id_map.txt'
# 声码器模型路径
voc_wavegan_model_path = 'models/wavegan/model'
# 获取文本前端
frontend_fastspeech2 = Frontend(g2p_model='g2pM', phone_vocab_path=fastspeech2_phones_dict_path)
# 声学模型
am_fastspeech2_inference = paddle.jit.load(am_fastspeech2_model_path)
# 声码器
voc_wavegan_inference = paddle.jit.load(voc_wavegan_model_path)
# 预热
am_fastspeech2_inference(paddle.ones((10,), dtype=paddle.int64))
voc_wavegan_inference(paddle.rand((86, 80), dtype=paddle.float32))


# ===========================效果好模型===================================
# 声学模型路径
am_tacotron2_model_path = 'models/tacotron2/model'
# 模型发声字典
tacotron2_phones_dict_path = 'models/tacotron2/phone_id_map.txt'
# 声码器模型路径
voc_hifigan_model_path = 'models/hifigan/model'
# 获取文本前端
frontend_tacotron2 = Frontend(g2p_model='g2pM', phone_vocab_path=tacotron2_phones_dict_path)
# 声学模型
am_tacotron2_inference = paddle.jit.load(am_tacotron2_model_path)
# 声码器
voc_hifigan_inference = paddle.jit.load(voc_hifigan_model_path)
# 预热
am_tacotron2_inference(paddle.ones((10,), dtype=paddle.int64))
voc_hifigan_inference(paddle.rand((86, 80), dtype=paddle.float32))

# 保存目录
save_audio_dir = 'output'
os.makedirs(save_audio_dir, exist_ok=True)


# 语音合成(速度快)
@app.route('/text2speech_fast', methods=['GET'])
def text2speech_fast():
    """
        语音合成(速度快)
        ---
        tags:
          - 语音相关
        parameters:
          - name: text
            in: query
            type: string
            required: true
            description: 合成的语音文本，长度建议不超过50个字

        responses:
          200:
            schema:
    """
    text = request.args.get('text')
    # 文本转模型输入
    input_ids = frontend_fastspeech2.get_input_ids(text, merge_sentences=False)
    phone_ids = input_ids['phone_ids']
    # 模型输出的拼接结果
    wav_all = None
    # 开始合成
    for i in range(len(phone_ids)):
        part_phone_ids = phone_ids[i]
        # 获取声学模型的输出
        mel = am_fastspeech2_inference(part_phone_ids)
        # 获取声码器模型输出
        wav = voc_wavegan_inference(mel)
        # 如果是第一次生成就不用拼接数据
        if wav_all is None:
            wav_all = wav
        else:
            wav_all = paddle.concat([wav_all, wav])

    # Tensor转Numpy
    wav = wav_all.numpy()
    output_path = os.path.join(save_audio_dir, f'{int(time.time() * 1000)}.wav')
    sf.write(output_path, wav, samplerate=24000)
    print(f'音频已保存在：{output_path}')

    return send_file(output_path,
                     mimetype='audio/wav',
                     as_attachment=True,
                     download_name=os.path.basename(output_path))


# 语音合成(效果好)
@app.route('/text2speech_well', methods=['GET'])
def text2speech_well():
    """
        语音合成(效果好)
        ---
        tags:
          - 语音相关
        parameters:
          - name: text
            in: query
            type: string
            required: true
            description: 合成的语音文本，长度建议不超过50个字

        responses:
          200:
            schema:
    """
    text = request.args.get('text')
    # 文本转模型输入
    input_ids = frontend_tacotron2.get_input_ids(text)
    phone_ids = input_ids['phone_ids']
    # 模型输出的拼接结果
    wav_all = None
    # 开始合成
    for i in range(len(phone_ids)):
        part_phone_ids = phone_ids[i]
        # 获取声学模型的输出
        mel = am_tacotron2_inference(part_phone_ids)
        # 获取声码器模型输出
        wav = voc_hifigan_inference(mel)
        # 如果是第一次生成就不用拼接数据
        if wav_all is None:
            wav_all = wav
        else:
            wav_all = paddle.concat([wav_all, wav])

    # Tensor转Numpy
    wav = wav_all.numpy()
    output_path = os.path.join(save_audio_dir, f'{int(time.time() * 1000)}.wav')
    sf.write(output_path, wav, samplerate=24000)
    print(f'音频已保存在：{output_path}')

    return send_file(output_path,
                     mimetype='audio/wav',
                     as_attachment=True,
                     download_name=os.path.basename(output_path))


if __name__ == '__main__':
    app.run(host='0.0.0.0')
