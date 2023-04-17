from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

model = ORTModelForSpeechSeq2Seq.from_pretrained("/home/mithil/PycharmProjects/africa-2000audio/model/whisper-small-baseline")