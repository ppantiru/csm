from generator import load_csm_1b, Segment
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

speakers = [0, 1, 0, 0]
transcripts = [
    "I can't get the neurotoxin into your head any faster.",
    "I'll use lasers to inscribe a line down the center of the facility, and one half will be where you live, and I'll live in the other half.",
    "It also says you are adopted, so that's funny too.",
    "It says so right here in your personnel file: unlikable, liked by no one, a bitter unlikable owner whose passing shall not be mourned.",
    "The rocket really is the way to go.",
    "Well, you've managed to destroy that part of me."
]
audio_paths = [
    "a1.wav",
    "a2.wav",
    "a3.wav",
    "a4.wav",
    "a5.wav",
    "a6.wav"
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="It says so right here in your personnel file: unlikable, liked by no one, a bitter unlikable owner whose passing shall not be mourned.",
    speaker=1,
    context=segments,
    max_audio_length_ms=30_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)