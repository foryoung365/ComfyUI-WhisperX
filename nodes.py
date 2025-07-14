import os
import srt
import torch
import time
import whisperx
import folder_paths
import cuda_malloc
import translators as ts
from tqdm import tqdm
from datetime import timedelta
import textwrap

input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()

class PreViewSRT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"srt": ("SRT",)}
                }

    CATEGORY = "AIFSH_WhisperX"

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    
    FUNCTION = "show_srt"

    def show_srt(self, srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, 'r', encoding="utf-8") as f:
            srt_content = f.read()
        return {"ui": {"srt":[srt_content,srt_name,dir_name]}}


class SRTToString:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"srt": ("SRT",)}
                }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "read"

    CATEGORY = "AIFSH_FishSpeech"

    def read(self,srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, 'r', encoding="utf-8") as f:
            srt_content = f.read()
        return (srt_content,)


class WhisperX:
    @classmethod
    def INPUT_TYPES(s):
        model_list = ["large-v3","distil-large-v3","large-v2", "large-v3-turbo"]
        translator_list = ['alibaba', 'apertium', 'argos', 'baidu', 'bing',
        'caiyun', 'cloudTranslation', 'deepl', 'elia', 'google',
        'hujiang', 'iciba', 'iflytek', 'iflyrec', 'itranslate',
        'judic', 'languageWire', 'lingvanex', 'mglip', 'mirai',
        'modernMt', 'myMemory', 'niutrans', 'papago', 'qqFanyi',
        'qqTranSmart', 'reverso', 'sogou', 'sysTran', 'tilde',
        'translateCom', 'translateMe', 'utibet', 'volcEngine', 'yandex',
        'yeekit', 'youdao']
        lang_list = ["zh","en","ja","ko","ru","fr","de","es","pt","it","ar"]
        format_list = ["srt", "txt"]
        return {"required":
                    {"audio": ("AUDIOPATH",),
                     "model_type":(model_list,{"default": "large-v3"}),
                     "batch_size":("INT",{"default": 4}),
                     "chunk_size": ("INT", {"default": 0, "min": 0}),
                     "output_format": (format_list, {"default": "srt"}),
                     "max_line_width": ("INT", {"default": 0, "min": 0}),
                     "initial_prompt": ("STRING", {"default": "", "multiline": True}),
                     "if_mutiple_speaker":("BOOLEAN",{"default": False}),
                     "use_auth_token":("STRING",{"default": "put your huggingface user auth token here for Assign speaker labels"}),
                     "if_translate":("BOOLEAN",{"default": False}),
                     "translator":(translator_list,{"default": "alibaba"}),
                     "to_language":(lang_list,{"default": "en"})
                     }
                }

    CATEGORY = "AIFSH_WhisperX"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("original_output_path", "translated_output_path")
    FUNCTION = "get_transcript"

    def _generate_output(self, result, output_format, max_line_width):
        if output_format == 'txt':
            lines = []
            for segment in result["segments"]:
                text = segment['text'].strip()
                if 'speaker' in segment:
                    lines.append(f"[{segment['speaker']}]: {text}")
                else:
                    lines.append(text)
            return "\n".join(lines)
        
        elif output_format == 'srt':
            subs = []
            for i, segment in enumerate(result["segments"]):
                start_time = timedelta(seconds=segment['start'])
                end_time = timedelta(seconds=segment['end'])
                
                text = segment['text'].strip()
                if 'speaker' in segment:
                    text = f"[{segment['speaker']}]: {text}"

                if max_line_width > 0:
                    text = '\n'.join(textwrap.wrap(text, width=max_line_width, break_long_words=False, replace_whitespace=False))

                subs.append(srt.Subtitle(index=i+1, start=start_time, end=end_time, content=text))
            return srt.compose(subs)
        return ""

    def get_transcript(self, audio, model_type, batch_size, chunk_size, output_format, max_line_width, initial_prompt, if_mutiple_speaker,
                use_auth_token, if_translate, translator, to_language):
        
        compute_type = "float16"
        device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
        
        audio_path = audio
        base_name = os.path.splitext(os.path.basename(audio_path))[0]

        # 1. Transcribe
        if model_type == "large-v3-turbo":
            model_type = "deepdml/faster-whisper-large-v3-turbo-ct2"
        
        asr_options = {}
        if initial_prompt:
            asr_options["initial_prompt"] = initial_prompt
            
        model = whisperx.load_model(model_type, device, compute_type=compute_type, asr_options=asr_options)
        audio_data = whisperx.load_audio(audio_path)
        
        transcribe_args = {"batch_size": batch_size}
        if chunk_size > 0:
            transcribe_args["chunk_size"] = chunk_size

        result = model.transcribe(audio_data, **transcribe_args)
        language_code = result["language"]
        
        # 2. Align
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio_data, device, return_char_alignments=False)
        
        # 3. Diarize
        if if_mutiple_speaker:
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=use_auth_token, device=device)
            diarize_segments = diarize_model(audio_data)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        # 4. Write original transcript
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        original_output_path = os.path.join(out_path, f"{timestamp}_{base_name}.{output_format}")
        original_content = self._generate_output(result, output_format, max_line_width)
        with open(original_output_path, 'w', encoding="utf-8") as f:
            f.write(original_content)

        # 5. Write translated transcript
        translated_output_path = original_output_path
        if if_translate:
            import copy
            translated_result = copy.deepcopy(result)
            for segment in tqdm(translated_result["segments"], desc="Translating ..."):
                try:
                    segment['text'] = ts.translate_text(query_text=segment['text'], translator=translator, to_language=to_language)
                except Exception as e:
                    print(f"Translation failed for segment: {segment['text']}. Error: {e}")

            translated_output_path = os.path.join(out_path, f"{timestamp}_{base_name}_{to_language}.{output_format}")
            translated_content = self._generate_output(translated_result, output_format, max_line_width)
            with open(translated_output_path, 'w', encoding="utf-8") as f:
                f.write(translated_content)

        # Cleanup
        import gc; gc.collect(); torch.cuda.empty_cache()
        
        return (original_output_path, translated_output_path)

class LoadAudioPath:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3","WAV","flac","m4a", "mp4"]]
        return {"required":
                    {"audio": (sorted(files),)}
                }

    CATEGORY = "AIFSH_WhisperX"

    RETURN_TYPES = ("AUDIOPATH",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)

class PathToAudioPath:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": "X:\\path\\to\\your\\audio_or_video.mp4"}),
            }
        }

    CATEGORY = "AIFSH_WhisperX"
    RETURN_TYPES = ("AUDIOPATH",)
    FUNCTION = "convert_path"

    def convert_path(self, path):
        return (path,)

class PathToSRT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "AIFSH_WhisperX"
    RETURN_TYPES = ("SRT",)
    FUNCTION = "convert_path"

    def convert_path(self, path):
        return (path,)
