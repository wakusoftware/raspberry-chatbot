import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pygame import mixer

load_dotenv()

client = OpenAI()


def stream_text():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=512)
    for chunk in llm.stream("Write me a song about sparkling water."):
        print(chunk.content, end="")


def stream_text_with_audio(prompt: str):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=1024)
    accumulated_text = ""  # Initialize an empty string to accumulate text
    for chunk in llm.stream(prompt):
        accumulated_text += chunk.content  # Accumulate text from each chunk
        print(chunk.content, end="")

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
                    tts(complete_sentence)  # Call tts with the complete sentence

            # Keep the last part if it's not a complete sentence, else reset to empty string
            accumulated_text = sentences[-1] if sentences[-1] else ""


def tts(text):
    speech_file_path = Path(__file__).parent / "speech.opus"
    response = client.audio.speech.create(
        model="tts-1", voice="shimmer", input=text, response_format="opus"
    )

    response.stream_to_file(speech_file_path)
    mixer.init()
    mixer.music.load(speech_file_path)
    mixer.music.play()
    while mixer.music.get_busy():  # wait for music to finish playing
        time.sleep(1)


if __name__ == "__main__":
    stream_text_with_audio("Who was Hugo Chavez?")
