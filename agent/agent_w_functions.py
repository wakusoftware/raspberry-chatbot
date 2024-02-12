import json
import os
import subprocess
import time
from pathlib import Path

import sounddevice as sd
from dotenv import load_dotenv
from openai import OpenAI
from pygame import mixer

load_dotenv()


# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


class AiAgent:
    """
    A class representing an AI agent capable of performing various tasks such as text-to-speech, audio recording,
    text streaming, and integrating with OpenAI's API for audio transcription and chat completions.

    Attributes:
        client (OpenAI): An instance of the OpenAI API client initialized with an API key.
        messages (list): A list to hold messages for processing.
        model (str): The model used for chat completions, defaulting to "gpt-3.5-turbo".
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        messages_limit: int = 12,
        system_prompt: str = "You are a helpful assistant. Always finish your sentences with a '.'",
    ):
        """
        Initializes the AiAgent with a specified model and an OpenAI client using the OPENAI_API_KEY environment variable.

        Parameters:
            model (str): The name of the model to use for chat completions. Defaults to "gpt-3.5-turbo".
        """
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.system_prompt = system_prompt
        self.messages = []
        self.messages.append(
            {
                "role": "system",
                "content": self.system_prompt,
            }
        )
        self.model = model
        self.messages_limit = messages_limit
        self.tools = None
        self.available_functions = None

    def add_tools(self, tools):
        """
        Adds a tool to the list of tools to be used for chat completions.

        Parameters:
            :param tools: A list of tools to be added to the list of tools.
        """
        # self.tools.append(tool)
        self.tools = tools
        self.available_functions = {
            "get_current_weather": get_current_weather,
        }
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

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

    def audio_interaction(
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

    def call_ai_function(self, function_name: str, function_arguments: str):
        # convert function_arguments to a python object
        function_arguments = json.loads(function_arguments)

        match function_name:

            case "get_current_weather":
                if not function_arguments["location"]:
                    raise ValueError("Location is required for get_current_weather")
                unit = function_arguments.get("unit") or "fahrenheit"
                response = get_current_weather(function_arguments["location"], unit)
                return response

    def stream_text_with_audio(self, user_input: str):
        """
        Streams text responses from the AI model and uses text-to-speech to audibly play back complete sentences.
        It accumulates text from streamed responses and checks for complete sentences to convert to speech.

        Parameters:
            user_input (str): The text input from the user to get responses for.
        """
        accumulated_text = ""  # Initialize an empty string to accumulate text# Add the user input to the messages array
        complete_answer = ""
        function_name = ""
        function_arguments = ""
        tool_call_id = ""
        self.messages.append({"role": "user", "content": user_input})

        # Ensure messages do not exceed the limit, preserving the system message at index 0
        if (
            len(self.messages) > self.messages_limit + 1
        ):  # +1 to account for the system message
            # Delete the 1st and 2nd user messages, preserving system message at index 0
            del self.messages[1:3]

        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=0,
            stream=True,  # again, we set stream=True
            tools=self.tools,
        ):

            print("chunk choices: ", chunk.choices)
            print("choices length: ", len(chunk.choices))
            print("finish reason: ", chunk.choices[0].finish_reason)

            message = chunk.choices[0].delta.content
            tool_calls = chunk.choices[0].delta.tool_calls

            if tool_calls is not None:
                print("message for function: ", chunk.choices[0].delta)

                if len(function_name) == 0:
                    function_name = tool_calls[0].function.name
                    tool_call_id = tool_calls[0].id

                function_arguments += tool_calls[0].function.arguments

            if chunk.choices[0].finish_reason == "tool_calls":
                function_response = self.call_ai_function(
                    function_name, function_arguments
                )
                print("function_response: ", function_response)
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "function": {
                                    "name": function_name,
                                    "arguments": function_arguments,
                                },
                                "type": "function",
                            }
                        ],
                    }
                )
                self.messages.append(
                    {
                        "tool_call_id": tool_call_id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
                tool_call_id = ""
                function_name = ""
                function_arguments = ""
                print("messages so far: ", self.messages)

                # Second call after the function response
                for chunk in self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=0,
                    stream=True,  # again, we set stream=True
                    # tools=self.tools,
                ):

                    message = chunk.choices[0].delta.content

                    if message is not None:
                        accumulated_text += message  # Accumulate text from each chunk
                        complete_answer += message  # Accumulate text from each chunk
                        print("mensaje: ", message, end="")

                        # Check if the accumulated text contains a complete sentence
                        if any(punct in accumulated_text for punct in ".!?"):
                            self.tts(accumulated_text)
                            accumulated_text = ""

                # Add the AI response to the messages array
                print("complete_answer: ", complete_answer)
                self.messages.append({"role": "assistant", "content": complete_answer})

            if message is not None:
                accumulated_text += message  # Accumulate text from each chunk
                complete_answer += message  # Accumulate text from each chunk
                # print("mensaje: ", message, end="")

                # Check if the accumulated text contains a complete sentence
                if any(punct in accumulated_text for punct in ".!?"):
                    self.tts(accumulated_text)
                    accumulated_text = ""

        # Add the AI response to the messages array
        # print("complete_answer: ", complete_answer)
        self.messages.append({"role": "assistant", "content": complete_answer})


if __name__ == "__main__":
    agent = AiAgent()
    agent.add_tools(["get_current_weather"])
    print(agent.tools)
    # agent.text_streaming(
    #     "tell me a story about a dragon and a knight who becomes friends."
    # )
    # agent.tts("I am a dragon and I am your friend.")
    # agent.stream_text_with_audio("Who was Bolivar?")
    agent.audio_interaction("output.mp3")
