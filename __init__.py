import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
WEB_DIRECTORY = "./web"
from .nodes import LoadAudioPath,WhisperX,PreViewSRT,SRTToString,PathToAudioPath,PathToSRT

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "LoadAudioVideoPath": LoadAudioPath,
    "WhisperX": WhisperX,
    "PreViewSRT":PreViewSRT,
    "SRTToString":SRTToString,
    "PathToAudioPath": PathToAudioPath,
    "PathToSRT": PathToSRT
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudioVideoPath": "LoadAudioVideoPath",
    "WhisperX": "WhisperX Node",
    "PreViewSRT":"PreView SRT",
    "SRTToString": "SRT to String",
    "PathToAudioPath": "Path to AudioPath",
    "PathToSRT": "Path to SRT"
}