import os, json, threading, time, subprocess, tempfile
from jnius import autoclass, PythonJavaClass, java_method
from PIL import Image as PILImage
import torch
from transformers import (
    AutoProcessor, AutoModelForCausalLM,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
import vosk
from threading import Event

# ---------- НАСТРОЙКИ ----------
WAKE_WORD = "слушай"
CAMERA_ID = 0  # 0 – задняя камера, 1 – фронтальная
MODEL_DIR = os.path.join(os.path.dirname(__file__), "assets", "models", "florence2")
TRANSLATE_MODEL = os.path.join(os.path.dirname(__file__), "assets", "models", "nllb-200")
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "assets", "models", "vosk-model-small-ru-0.22")

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

# ---------- Глобальные переменные ----------
florence_processor = None
florence_model = None
translator_tokenizer_ru_en = None
translator_model = None
translator_tokenizer_en_ru = None
vosk_recognizer = None
tts = None

# ---------- TTS ----------
def init_tts():
    global tts
    tts = TextToSpeech(activity, None)

def speak(text):
    if tts is None:
        init_tts()
    tts.speak(text, TextToSpeech.QUEUE_FLUSH, None)

# ---------- Vosk ----------
def init_vosk():
    global vosk_recognizer
    vosk_model = vosk.Model(VOSK_MODEL_PATH)
    vosk_recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

def record_audio(duration=3):
    sample_rate = 16000
    channel_config = AudioFormat.CHANNEL_IN_MONO
    audio_format = AudioFormat.ENCODING_PCM_16BIT
    buffer_size = AudioRecord.getMinBufferSize(sample_rate, channel_config, audio_format)

    recorder = AudioRecord(
        MediaRecorder.AudioSource.MIC,
        sample_rate,
        channel_config,
        audio_format,
        buffer_size
    )
    recorder.startRecording()
    data = bytearray()
    num_samples = int(sample_rate * duration)
    chunk = buffer_size // 2
    for _ in range(0, num_samples, chunk):
        buf = bytearray(chunk)
        read = recorder.read(buf, 0, chunk)
        if read > 0:
            data.extend(buf[:read])
    recorder.stop()
    recorder.release()
    return bytes(data)

def listen():
    raw = record_audio(3)
    if vosk_recognizer.AcceptWaveform(raw):
        result = json.loads(vosk_recognizer.Result())
        text = result.get("text", "").strip().lower()
        return text if text else None
    return None

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
    global translator_tokenizer_ru_en, translator_model, translator_tokenizer_en_ru

    florence_processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
    florence_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.float32, low_cpu_mem_usage=True)

    translator_tokenizer_ru_en = AutoTokenizer.from_pretrained(
        TRANSLATE_MODEL, src_lang="rus_Cyrl", tgt_lang="eng_Latn")
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATE_MODEL)
    translator_tokenizer_en_ru = AutoTokenizer.from_pretrained(
        TRANSLATE_MODEL, src_lang="eng_Latn", tgt_lang="rus_Cyrl")

def translate_ru2en(text):
    inputs = translator_tokenizer_ru_en(text, return_tensors="pt", padding=True)
    out = translator_model.generate(**inputs, max_length=200)
    return translator_tokenizer_ru_en.decode(out[0], skip_special_tokens=True)

def translate_en2ru(text):
    inputs = translator_tokenizer_en_ru(text, return_tensors="pt", padding=True)
    out = translator_model.generate(**inputs, max_length=200)
    return translator_tokenizer_en_ru.decode(out[0], skip_special_tokens=True)

def answer_question(image, question_en):
    inputs = florence_processor(text=question_en, images=image, return_tensors="pt")
    generated_ids = florence_model.generate(
        **inputs, max_new_tokens=200, num_beams=3, early_stopping=True)
    return florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ---------- Главный цикл ----------
def main():
    init_tts()
    init_vosk()
    load_models()
    speak("Ассистент запущен")

    while True:
        print("Waiting for wake word...")
        text = listen()
        if text and WAKE_WORD in text:
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