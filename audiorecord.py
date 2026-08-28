from jnius import autoclass

AudioRecord = autoclass('android.media.AudioRecord')
AudioFormat = autoclass('android.media.AudioFormat')
MediaRecorder = autoclass('android.media.MediaRecorder')

SAMPLE_RATE = 16000
CHANNEL = AudioFormat.CHANNEL_IN_MONO
ENCODING = AudioFormat.ENCODING_PCM_16BIT
BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)

class AudioRecorder:
    def record(self, duration=3):
        recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL,
            ENCODING,
            BUFFER_SIZE
        )
        recorder.startRecording()
        frames = []
        num_samples = int(SAMPLE_RATE * duration)
        for _ in range(0, num_samples, BUFFER_SIZE):
            data = bytearray(BUFFER_SIZE)
            bytes_read = recorder.read(data, 0, BUFFER_SIZE)
            if bytes_read > 0:
                frames.append(bytes(data[:bytes_read]))
        recorder.stop()
        recorder.release()
        return b''.join(frames)