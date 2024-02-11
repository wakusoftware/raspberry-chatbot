import subprocess

import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI

from langchain_audio import stream_text_with_audio

load_dotenv()

client = OpenAI()


def record_audio_mp3_on_command(
    filename_template, sample_rate=44100, channels=1, bitrate="64k"
):
    session_count = 1
    while True:
        # Configure ffmpeg command for MP3 encoding with dynamic filename
        filename = filename_template.format(session_count)
        ffmpeg_command = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-f",
            "s16le",  # Input format is signed 16-bit little endian samples
            "-ar",
            str(sample_rate),  # Set input sample rate
            "-ac",
            str(channels),  # Set number of input channels
            "-i",
            "-",  # Indicate that input comes from stdin
            "-codec:a",
            "libmp3lame",  # Specify audio codec
            "-b:a",
            bitrate,  # Specify bitrate for better quality
            filename,  # Output filename
        ]

        print("Press Enter to start recording or type 'exit' to quit:")
        command = input()  # Wait for the user to press Enter or type exit

        if command.lower() == "exit":
            break  # Exit the loop and end the program

        # Start ffmpeg process for recording
        ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

        # Define callback function to feed audio to ffmpeg
        def callback(indata, frames, time, status):
            ffmpeg_process.stdin.write(indata)

        print(f"Recording session {session_count}... Press Enter to stop.")

        # Start recording
        stream = sd.InputStream(
            samplerate=sample_rate, channels=channels, dtype="int16", callback=callback
        )
        stream.start()

        input()  # Wait for the user to press Enter to stop recording

        # Stop recording
        stream.stop()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        print(f"Recording session {session_count} stopped.")

        audio_file = open("output.mp3", "rb")

        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text"
        )

        # chat_completion_streaming(transcript)
        stream_text_with_audio(transcript)

        # print("Transcript: ", transcript)

        session_count += 1  # Increment session count for next filename


if __name__ == "__main__":
    record_audio_mp3_on_command("output.mp3")
    # record_audio_mp3_on_command("session_{}.mp3")
