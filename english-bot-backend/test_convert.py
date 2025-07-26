from pydub import AudioSegment

# Altere aqui para um áudio que você salvou do WhatsApp
ogg_file = "teste.ogg"
wav_file = "teste.wav"

audio = AudioSegment.from_file(ogg_file, format="ogg")
audio.export(wav_file, format="wav")

print("✅ Arquivo convertido com sucesso.") 