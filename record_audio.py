import wave

import pyaudio
from pydub import AudioSegment


def record_audio_wav(
    filename,
    duration=5,
    sample_rate=44100,
    chunk_size=1024,
    channels=1,
    format=pyaudio.paInt16,
):
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=format,
        channels=channels,
        rate=sample_rate,
        input=True,
        frames_per_buffer=chunk_size,
    )
    print(f"Recording for {duration} seconds...")
    frames = []
    for _ in range(int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)
    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))


def convert_wav_to_mp3(wav_filename, mp3_filename):
    sound = AudioSegment.from_wav(wav_filename)
    sound.export(mp3_filename, format="mp3")


if __name__ == "__main__":
    wav_filename = "output.wav"
    mp3_filename = "output.mp3"
    record_audio_wav(wav_filename, duration=5)
    convert_wav_to_mp3(wav_filename, mp3_filename)
