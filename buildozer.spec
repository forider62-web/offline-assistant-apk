[app]
title = Offline Assistant
package.name = offlineassistant
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,json,txt,bin,spm,model,safetensors,dict,lm
version = 1.0
requirements = python3, kivy, pillow, transformers, torch, vosk, sentencepiece, pydub
android.permissions = CAMERA, RECORD_AUDIO, INTERNET
android.api = 33
android.minapi = 24
android.ndk = 25b
android.sdk = 34
android.arch = arm64-v8a
p4a.branch = develop