import os, json, threading, time, struct
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.core.camera import Camera as KivyCamera
from PIL import Image as PILImage
import torch
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
import vosk
from audiorecord import AudioRecorder

# Глобальные модели
florence_processor = None
florence_model = None
translator_tokenizer_ru_en = None
translator_model = None
translator_tokenizer_en_ru = None
vosk_recognizer = None

def load_models():
    global florence_processor, florence_model
    global translator_tokenizer_ru_en, translator_model, translator_tokenizer_en_ru
    global vosk_recognizer

    # Пути к моделям (скачаются в папку assets/models на сервере сборки)
    base = os.path.join(os.path.dirname(__file__), "assets", "models")

    florence_processor = AutoProcessor.from_pretrained(
        os.path.join(base, "florence2"), trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(
        os.path.join(base, "florence2"), trust_remote_code=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True)

    nllb_path = os.path.join(base, "nllb-200")
    translator_tokenizer_ru_en = AutoTokenizer.from_pretrained(
        nllb_path, src_lang="rus_Cyrl", tgt_lang="eng_Latn")
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_path)
    translator_tokenizer_en_ru = AutoTokenizer.from_pretrained(
        nllb_path, src_lang="eng_Latn", tgt_lang="rus_Cyrl")

    vosk_model = vosk.Model(os.path.join(base, "vosk-model-small-ru-0.22"))
    vosk_recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
    print("Models loaded")

def translate_ru2en(text):
    inputs = translator_tokenizer_ru_en(text, return_tensors="pt", padding=True)
    out = translator_model.generate(**inputs, max_length=200)
    return translator_tokenizer_ru_en.decode(out[0], skip_special_tokens=True)

def translate_en2ru(text):
    inputs = translator_tokenizer_en_ru(text, return_tensors="pt", padding=True)
    out = translator_model.generate(**inputs, max_length=200)
    return translator_tokenizer_en_ru.decode(out[0], skip_special_tokens=True)

def get_answer(image, question_en):
    inputs = florence_processor(text=question_en, images=image, return_tensors="pt")
    generated_ids = florence_model.generate(
        **inputs, max_new_tokens=200, num_beams=3, early_stopping=True)
    return florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

class AssistantApp(App):
    def build(self):
        self.recorder = AudioRecorder()
        self.tts = None

        layout = BoxLayout(orientation='vertical')
        self.camera = KivyCamera(play=True, resolution=(640, 480))
        layout.add_widget(self.camera)

        self.status_label = Label(text="Загрузка моделей...")
        layout.add_widget(self.status_label)

        self.btn = Button(text="Слушать (3 сек)", size_hint=(1, 0.15))
        self.btn.bind(on_press=self.capture_and_ask)
        layout.add_widget(self.btn)

        threading.Thread(target=load_models, daemon=True).start()
        Clock.schedule_interval(self.check_models_loaded, 1.0)
        return layout

    def check_models_loaded(self, dt):
        if florence_model is not None and vosk_recognizer is not None:
            self.status_label.text = "Готово. Скажите 'слушай' или нажмите кнопку"
            return False

    def capture_and_ask(self, instance):
        if florence_model is None:
            self.status_label.text = "Модели ещё загружаются..."
            return
        self.status_label.text = "Слушаю..."
        self.btn.disabled = True
        threading.Thread(target=self.record_and_ask, daemon=True).start()

    def record_and_ask(self):
        raw_data = self.recorder.record(duration=3)
        if vosk_recognizer.AcceptWaveform(raw_data):
            result = json.loads(vosk_recognizer.Result())
            text = result.get("text", "").strip().lower()
        else:
            text = ""
        Clock.schedule_once(lambda dt: self.handle_voice(text))

    def handle_voice(self, text):
        self.btn.disabled = False
        if not text:
            self.status_label.text = "Ничего не услышано"
            return
        self.status_label.text = f"Распознано: {text}"
        if "слушай" in text:
            self.status_label.text = "Говорите вопрос..."
            threading.Thread(target=self.record_question, daemon=True).start()
        else:
            self.status_label.text = "Не похоже на команду 'слушай'"

    def record_question(self):
        raw_data = self.recorder.record(duration=5)
        if vosk_recognizer.AcceptWaveform(raw_data):
            result = json.loads(vosk_recognizer.Result())
            question_ru = result.get("text", "").strip()
        else:
            question_ru = ""
        Clock.schedule_once(lambda dt: self.answer_question(question_ru))

    def answer_question(self, question_ru):
        if not question_ru:
            self.status_label.text = "Вопрос не распознан"
            return
        self.status_label.text = "Обработка..."
        frame = self.get_current_frame()
        if frame is None:
            self.status_label.text = "Не удалось получить изображение"
            return
        try:
            question_en = translate_ru2en(question_ru)
            answer_en = get_answer(frame, question_en)
            answer_ru = translate_en2ru(answer_en)
            self.status_label.text = answer_ru
            self.speak(answer_ru)
        except Exception as e:
            self.status_label.text = f"Ошибка: {e}"

    def get_current_frame(self):
        if self.camera.texture is not None:
            pixels = self.camera.texture.pixels
            size = self.camera.texture.size
            img = PILImage.frombytes(mode='RGBA', size=size, data=pixels)
            return img.convert('RGB')
        return None

    def speak(self, text):
        from jnius import autoclass
        if self.tts is None:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            activity = PythonActivity.mActivity
            TextToSpeech = autoclass('android.speech.tts.TextToSpeech')
            self.tts = TextToSpeech(activity, None)
        self.tts.speak(text, 0, None)

if __name__ == '__main__':
    AssistantApp().run()