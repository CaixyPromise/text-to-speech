import os
import time
import warnings
import wave
import tkinter.messagebox
from tkinter import *

import paddle
import pyaudio
import soundfile as sf
import threading

warnings.filterwarnings("ignore")
from paddlespeech.t2s.frontend.zh_frontend import Frontend


class MysApp:
    def __init__(self, window: Tk):
        # ===========================快速模型===================================
        # 声学模型路径
        self.am_fastspeech2_model_path = 'models/fastspeech2/model'
        # 模型发声字典
        self.fastspeech2_phones_dict_path = 'models/fastspeech2/phone_id_map.txt'
        # 声码器模型路径
        self.voc_wavegan_model_path = 'models/wavegan/model'
        # 获取文本前端
        self.frontend_fastspeech2 = Frontend(g2p_model='g2pM', phone_vocab_path=self.fastspeech2_phones_dict_path)
        # 声学模型
        self.am_fastspeech2_inference = paddle.jit.load(self.am_fastspeech2_model_path)
        # 声码器
        self.voc_wavegan_inference = paddle.jit.load(self.voc_wavegan_model_path)
        # 预热
        self.am_fastspeech2_inference(paddle.ones((10,), dtype=paddle.int64))
        self.voc_wavegan_inference(paddle.rand((86, 80), dtype=paddle.float32))

        # ===========================效果好模型===================================
        # 声学模型路径
        self.am_tacotron2_model_path = 'models/tacotron2/model'
        # 模型发声字典
        self.tacotron2_phones_dict_path = 'models/tacotron2/phone_id_map.txt'
        # 声码器模型路径
        self.voc_hifigan_model_path = 'models/hifigan/model'
        # 获取文本前端
        self.frontend_tacotron2 = Frontend(g2p_model='g2pM', phone_vocab_path=self.tacotron2_phones_dict_path)
        # 声学模型
        self.am_tacotron2_inference = paddle.jit.load(self.am_tacotron2_model_path)
        # 声码器
        self.voc_hifigan_inference = paddle.jit.load(self.voc_hifigan_model_path)
        # 预热
        self.am_tacotron2_inference(paddle.ones((10,), dtype=paddle.int64))
        self.voc_hifigan_inference(paddle.rand((86, 80), dtype=paddle.float32))

        # 保存目录
        self.save_audio_dir = 'output'
        os.makedirs(self.save_audio_dir, exist_ok=True)
        # 播放器
        self.p = pyaudio.PyAudio()
        self.fast_model = True

        self.running = False
        self.window = window
        # 指定窗口标题
        self.window.title("夜雨飘零语音合成工具")
        # 固定窗口大小
        self.window.geometry('900x280')
        self.window.resizable(False, False)
        label1 = Label(self.window, text="合成文本：")
        label1.place(x=10, y=10)
        self.input_text = Text(self.window, width=80, height=8)
        self.input_text.place(x=90, y=10)
        self.button2 = Button(self.window, text="合成", width=15, command=self.text2speech_btn)
        self.button2.place(x=700, y=20)
        self.button3 = Button(self.window, text="合成并播放", width=15, command=self.text2speech_play_btn)
        self.button3.place(x=700, y=70)
        label1 = Label(self.window, text="输出日志：")
        label1.place(x=10, y=130)
        self.output_text = Text(self.window, width=100, height=8)
        self.output_text.place(x=10, y=160)
        # 选择模型
        self.check_var = BooleanVar()
        self.check_var.set(True)
        self.fast_model_check = Checkbutton(self.window, text='是否使用快速模型，否则使用效果好的模型', variable=self.check_var,
                                            command=self.select_model)
        self.fast_model_check.pack()
        self.fast_model_check.place(x=600, y=130)

    # 改变是否开启自动标注
    def select_model(self):
        self.fast_model = self.check_var.get()
        self.output_text.delete('1.0', END)
        if self.fast_model:
            self.output_text.insert(END, '使用快速模型\n')
        else:
            self.output_text.insert(END, '使用效果好模型\n')

    # 合成
    def text2speech_btn(self):
        thread = threading.Thread(target=self.text2speech, args=())
        thread.start()

    # 合成并播放
    def text2speech_play_btn(self):
        thread = threading.Thread(target=self.text2speech_play_thread, args=())
        thread.start()

    def text2speech_play_thread(self):
        audio_path = self.text2speech()
        if audio_path is not None:
            self.play(audio_path)

    # 合成语音
    def text2speech(self):
        if self.running:
            tkinter.messagebox.showwarning('警告', message='程序正在运行，请稍等！')
            return None
        self.running = True
        # 清空输出框
        self.output_text.delete('1.0', END)
        # 获取合成文本
        text = self.input_text.get('1.0', END).replace('\n', '。')
        self.output_text.insert(END, '合成文本为：{}\n'.format(text))
        # 文本转模型输入
        if self.fast_model:
            input_ids = self.frontend_fastspeech2.get_input_ids(text, merge_sentences=False)
        else:
            input_ids = self.frontend_tacotron2.get_input_ids(text)
        phone_ids = input_ids['phone_ids']
        # 模型输出的拼接结果
        wav_all = None
        # 开始合成
        for i in range(len(phone_ids)):
            part_phone_ids = phone_ids[i]
            # 获取声学模型的输出
            if self.fast_model:
                mel = self.am_fastspeech2_inference(part_phone_ids)
            else:
                mel = self.am_tacotron2_inference(part_phone_ids)
            # 获取声码器模型输出
            if self.fast_model:
                wav = self.voc_wavegan_inference(mel)
            else:
                wav = self.voc_hifigan_inference(mel)
            # 如果是第一次生成就不用拼接数据
            if wav_all is None:
                wav_all = wav
            else:
                wav_all = paddle.concat([wav_all, wav])

        # Tensor转Numpy
        wav = wav_all.numpy()
        output_path = os.path.join(self.save_audio_dir, f'{int(time.time() * 1000)}.wav')
        sf.write(output_path, wav, samplerate=24000)
        self.output_text.insert(END, '合成语音文件保存在：{}\n'.format(output_path))
        self.running = False
        return output_path

    # 播放
    def play(self, audio_path):
        self.running = True
        chunk = 1024
        wf = wave.open(audio_path, 'rb')
        stream = self.p.open(format=self.p.get_format_from_width(wf.getsampwidth()),
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
        self.running = False


tk = Tk()
myapp = MysApp(tk)

if __name__ == '__main__':
    tk.mainloop()
