import os
import subprocess
import time
from pathlib import Path

import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from pygame import mixer

load_dotenv()


class AiAgent:
    """
    A class representing an AI agent capable of performing various tasks such as text-to-speech, audio recording,
    text streaming, and integrating with OpenAI's API for audio transcription and chat completions.

    Attributes:
        client (OpenAI): An instance of the OpenAI API client initialized with an API key.
        messages (list): A list to hold messages for processing.
        model (str): The model used for chat completions, defaulting to "gpt-3.5-turbo".
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """
        Initializes the AiAgent with a specified model and an OpenAI client using the OPENAI_API_KEY environment variable.

        Parameters:
            model (str): The name of the model to use for chat completions. Defaults to "gpt-3.5-turbo".
        """
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.messages = []
        self.model = model

    def tts(self, text: str):
        """
        Converts text to speech using OpenAI's text-to-speech API and plays the audio.

        Parameters:
            text (str): The text to be converted to speech.
        """
        speech_file_path = Path(__file__).parent / "speech.opus"
        response = self.client.audio.speech.create(
            model="tts-1", voice="shimmer", input=text, response_format="opus"
        )

        response.stream_to_file(speech_file_path)
        mixer.init()
        mixer.music.load(speech_file_path)
        mixer.music.play()
        while mixer.music.get_busy():  # wait for music to finish playing
            time.sleep(1)

    def record_audio_mp3_on_command(
        self, filename_template, sample_rate=44100, channels=1, bitrate="64k"
    ):
        """
        Records audio in MP3 format on command from the user. The function remains in a loop, allowing for multiple
        recording sessions until the user decides to exit. It also transcribes the recorded audio using OpenAI's
        transcription API and streams text along with audio for complete sentences.

        Parameters:
            filename_template (str): A template for naming recorded audio files, with placeholders for session count.
            sample_rate (int): The sample rate for the audio recording. Defaults to 44100 Hz.
            channels (int): The number of audio channels. Defaults to 1 (mono).
            bitrate (str): The bitrate for the MP3 encoding. Defaults to "64k".
        """
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
                samplerate=sample_rate,
                channels=channels,
                dtype="int16",
                callback=callback,
            )
            stream.start()

            input()  # Wait for the user to press Enter to stop recording

            # Stop recording
            stream.stop()
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            print(f"Recording session {session_count} stopped.")

            audio_file = open("output.mp3", "rb")

            transcript = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )

            # chat_completion_streaming(transcript)
            self.stream_text_with_audio(transcript)

            # print("Transcript: ", transcript)

            session_count += 1  # Increment session count for next filename

    def stream_text(self, user_input: str):
        """
        Streams text responses from the AI model based on the user's input.

        Parameters:
            user_input (str): The text input from the user to get responses for.
        """
        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            temperature=0,
            stream=True,  # again, we set stream=True
        ):
            message = chunk.choices[0].delta.content
            if message is not None:
                print(message, end="")

    def stream_text_with_audio(self, user_input: str):
        """
        Streams text responses from the AI model and uses text-to-speech to audibly play back complete sentences.
        It accumulates text from streamed responses and checks for complete sentences to convert to speech.

        Parameters:
            user_input (str): The text input from the user to get responses for.
        """
        accumulated_text = ""  # Initialize an empty string to accumulate text
        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ],
            temperature=0,
            stream=True,  # again, we set stream=True
        ):
            message = chunk.choices[0].delta.content
            accumulated_text += message  # Accumulate text from each chunk
            print(message, end="")

            # Check if the accumulated text contains a complete sentence
            if "." in accumulated_text:
                # Split the accumulated text into sentences
                sentences = accumulated_text.split(".")
                for i, sentence in enumerate(
                    sentences[:-1]
                ):  # Exclude the last part if it's not a complete sentence
                    complete_sentence = (
                        sentence.strip() + "."
                    )  # Re-add the period to each complete sentence
                    if complete_sentence:  # Ensure the sentence is not empty
                        self.tts(
                            complete_sentence
                        )  # Call tts with the complete sentence

                # Keep the last part if it's not a complete sentence, else reset to empty string
                accumulated_text = sentences[-1] if sentences[-1] else ""


if __name__ == "__main__":
    agent = AiAgent()
    # agent.text_streaming(
    #     "tell me a story about a dragon and a knight who becomes friends."
    # )
    # agent.tts("I am a dragon and I am your friend.")
    # agent.stream_text_with_audio("Who was Bolivar?")
    agent.record_audio_mp3_on_command("output.mp3")
