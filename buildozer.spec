[app]
title = Offline Assistant
package.name = offlineassistant
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,jpeg,json,txt,bin,spm,model,safetensors,dict,lm,zip
version = 1.0
requirements = python3, pillow, transformers, torch, vosk, pyjnius, android
android.permissions = CAMERA, RECORD_AUDIO, INTERNET
android.api = 33
android.minapi = 24
android.ndk = 25b
android.sdk = 34
android.archs = arm64-v8a
android.bootstrap = webview
p4a.branch = develop