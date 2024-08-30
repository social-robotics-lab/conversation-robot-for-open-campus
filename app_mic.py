import argparse
import os
# import pytz
import time
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer
from threading import Event
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from robottools import RobotTools
from state_machine_mic import StateMachineThread


# コマンドライン引数
parse = argparse.ArgumentParser()
parse.add_argument("--ip", required=True)
parse.add_argument("--port", default=22222, type=int)
args = parse.parse_args()

# .env設定ファイル
load_dotenv()

# Azure Speech Recognition
speech_config = SpeechConfig(subscription=os.environ.get("AZURE_API_KEY"), region=os.environ.get("AZURE_SERVICE_REGION"))
speech_config.speech_recognition_language=os.environ.get("AZURE_SPEECH_RECOGNITION_LANGUAGE")

with (OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) as openai_client,
      MongoClient(os.environ.get("MONGO_HOST"), server_api=ServerApi('1')) as mongo_client):
    
    azure_asr_client = SpeechRecognizer(speech_config=speech_config)
    # RobotController
    robotcontroller_client = RobotTools(args.ip, args.port)
    # Client objects
    clients = dict(
        OPENAI_CLIENT=openai_client,
        MONGO_CLIENT=mongo_client,
        AZURE_ASR_CLIENT=azure_asr_client,
        ROBOTCONTROLLER_CLIENT=robotcontroller_client,
    )
    # StopEventObject for threads
    STOP_EVENT = Event()
    # StateMachineThread
    smt = StateMachineThread(clients, STOP_EVENT)
    smt.start()
    # Event loop
    try:
        while True:
            time.sleep(10)

    except KeyboardInterrupt:
        pass

    STOP_EVENT.set()

