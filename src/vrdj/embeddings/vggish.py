import torch
from torchvggish import vggish_input, vggish

vector_length = 128

class Model:
    def __init__(self, device = 'cpu'):
        self.device = torch.device(device)

    @property
    def model(self):
        if not hasattr(self, '_model'):
            model = vggish()
            model.eval()
            self._model = model.to(self.device)
        return self._model

    def waveform(self, filepath):
        return vggish_input.wavfile_to_examples(filepath)

    def embedding(self, audio):
        with torch.no_grad():
            if isinstance(audio, str):
                audio = self.waveform(audio)
            audio = audio.to(self.device)
            emb = self.model.forward(audio)
            return emb.cpu().numpy().astype('float32')
    
            
