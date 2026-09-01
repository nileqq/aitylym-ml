from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()

def tmstamps(wav, sampling_rate=16000):
    return get_speech_timestamps(wav, model, sampling_rate=sampling_rate, return_seconds=True)

