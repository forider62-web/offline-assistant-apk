import os, json, threading, time, subprocess, tempfile
from jnius import autoclass, PythonJavaClass, java_method
from PIL import Image as PILImage
import torch
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from threading import Event

# ---------- НАСТРОЙКИ ----------
WAKE_WORD = "слушай"
CAMERA_ID = 0
MODEL_DIR = os.path.join(os.path.dirname(__file__), "assets", "models", "florence2")
RU_EN_DIR = os.path.join(os.path.dirname(__file__), "assets", "models", "opus-mt-ru-en")
EN_RU_DIR = os.path.join(os.path.dirname(__file__), "assets", "models", "opus-mt-en-ru")

# ---------- Android классы ----------
PythonActivity = autoclass('org.kivy.android.PythonActivity')
activity = PythonActivity.mActivity
AudioRecord = autoclass('android.media.AudioRecord')
AudioFormat = autoclass('android.media.AudioFormat')
MediaRecorder = autoclass('android.media.MediaRecorder')
Camera = autoclass('android.hardware.Camera')
TextToSpeech = autoclass('android.speech.tts.TextToSpeech')
FileOutputStream = autoclass('java.io.FileOutputStream')
File = autoclass('java.io.File')
Environment = autoclass('android.os.Environment')
SpeechRecognizer = autoclass('android.speech.SpeechRecognizer')
RecognizerIntent = autoclass('android.content.Intent')
Bundle = autoclass('android.os.Bundle')
ArrayList = autoclass('java.util.ArrayList')

# ---------- Глобальные переменные ----------
florence_processor = None
florence_model = None
ru_en_tokenizer = None
ru_en_model = None
en_ru_tokenizer = None
en_ru_model = None
tts = None
speech_recognizer = None
recognition_result = None
recognition_event = Event()

# ---------- TTS ----------
def init_tts():
    global tts
    tts = TextToSpeech(activity, None)

def speak(text):
    if tts is None:
        init_tts()
    tts.speak(text, TextToSpeech.QUEUE_FLUSH, None)

# ---------- Speech Recognizer ----------
class RecognitionListener(PythonJavaClass):
    __javainterfaces__ = ['android/speech/RecognitionListener']
    __javacontext__ = 'app'

    @java_method('(I)V')
    def onReadyForSpeech(self, params):
        pass

    @java_method('(I)V')
    def onBeginningOfSpeech(self):
        pass

    @java_method('(F)V')
    def onRmsChanged(self, rmsdB):
        pass

    @java_method('([B)V')
    def onBufferReceived(self, buffer):
        pass

    @java_method('(I)V')
    def onEndOfSpeech(self):
        pass

    @java_method('(I)V')
    def onError(self, error):
        global recognition_result
        recognition_result = None
        recognition_event.set()

    @java_method('(Landroid/os/Bundle;)V')
    def onResults(self, results):
        global recognition_result
        matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
        if matches and len(matches) > 0:
            recognition_result = matches[0]
        else:
            recognition_result = None
        recognition_event.set()

    @java_method('(Landroid/os/Bundle;)V')
    def onPartialResults(self, partialResults):
        pass

    @java_method('(I)V')
    def onEvent(self, eventType, params):
        pass

def init_speech_recognizer():
    global speech_recognizer, recognition_listener
    recognition_listener = RecognitionListener()
    speech_recognizer = SpeechRecognizer.createSpeechRecognizer(activity)
    speech_recognizer.setRecognitionListener(recognition_listener)

def listen():
    global recognition_result, recognition_event
    recognition_result = None
    recognition_event.clear()

    intent = RecognizerIntent()
    intent.setAction(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
    intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
    intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ru-RU")
    intent.putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)

    speech_recognizer.startListening(intent)
    recognition_event.wait(timeout=10)
    speech_recognizer.stopListening()
    return recognition_result

# ---------- Камера ----------
class PictureCallback(PythonJavaClass):
    __javainterfaces__ = ['android/hardware/Camera$PictureCallback']
    __javacontext__ = 'app'

    def __init__(self, filepath, event):
        super().__init__()
        self.filepath = filepath
        self.event = event

    @java_method('([BLandroid/hardware/Camera;)V')
    def onPictureTaken(self, data, camera):
        try:
            fos = FileOutputStream(self.filepath)
            fos.write(data)
            fos.close()
        except Exception as e:
            print(f"Error saving photo: {e}")
        finally:
            self.event.set()

def take_photo():
    camera = None
    tmp_file = None
    try:
        camera = Camera.open(CAMERA_ID)
        params = camera.getParameters()
        params.setJpegQuality(85)
        camera.setParameters(params)
        camera.startPreview()

        tmp_file = os.path.join(tempfile.gettempdir(), f"photo_{int(time.time())}.jpg")
        event = Event()
        jpeg_callback = PictureCallback(tmp_file, event)
        camera.takePicture(None, None, jpeg_callback)

        if not event.wait(timeout=10):
            raise TimeoutError("Camera capture timed out")

        if not os.path.exists(tmp_file) or os.path.getsize(tmp_file) == 0:
            raise IOError("Failed to capture photo")

        return tmp_file
    except Exception as e:
        print(f"Camera error: {e}")
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
        return None
    finally:
        if camera:
            try:
                camera.stopPreview()
                camera.release()
            except:
                pass

# ---------- Загрузка моделей ----------
def load_models():
    global florence_processor, florence_model
    global ru_en_tokenizer, ru_en_model, en_ru_tokenizer, en_ru_model

    florence_processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True)

    ru_en_tokenizer = AutoTokenizer.from_pretrained(RU_EN_DIR)
    ru_en_model = AutoModelForSeq2SeqLM.from_pretrained(RU_EN_DIR)

    en_ru_tokenizer = AutoTokenizer.from_pretrained(EN_RU_DIR)
    en_ru_model = AutoModelForSeq2SeqLM.from_pretrained(EN_RU_DIR)

def translate_ru2en(text):
    inputs = ru_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    out = ru_en_model.generate(**inputs, max_length=200)
    return ru_en_tokenizer.decode(out[0], skip_special_tokens=True)

def translate_en2ru(text):
    inputs = en_ru_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    out = en_ru_model.generate(**inputs, max_length=200)
    return en_ru_tokenizer.decode(out[0], skip_special_tokens=True)

def answer_question(image, question_en):
    inputs = florence_processor(text=question_en, images=image, return_tensors="pt")
    generated_ids = florence_model.generate(
        **inputs, max_new_tokens=200, num_beams=3, early_stopping=True)
    return florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ---------- Главный цикл ----------
def main():
    init_tts()
    init_speech_recognizer()
    load_models()
    speak("Ассистент запущен")
    while True:
        text = listen()
        if text and WAKE_WORD in text.lower():
            speak("Слушаю")
            question = listen()
            if question:
                question_en = translate_ru2en(question)
                photo_path = take_photo()
                if photo_path:
                    img = PILImage.open(photo_path).convert("RGB")
                    os.unlink(photo_path)
                    ans_en = answer_question(img, question_en)
                    ans_ru = translate_en2ru(ans_en)
                    speak(ans_ru)
                else:
                    speak("Не удалось сделать снимок")
            else:
                speak("Не расслышала вопрос")
        time.sleep(0.5)

if __name__ == '__main__':
    main()