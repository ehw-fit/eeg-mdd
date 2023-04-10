import numpy as np

class GAops:
    def __init__(self, channels, freqs):
        self.channels = channels
        self.channels_max = len(channels)
        self.freqs = freqs
        self.freqs_max = len(freqs)
        self.fun_max = 5
        self.max_length = 30
        self.window_range = (4, 160)

    def limit_frequency(self, max_frequency):
        self.freqs_max = (int)(np.argwhere(np.array(self.freqs) <= max_frequency)[-1])
        print("Freqeuncy limited to {} Hz".format(self.freqs[self.freqs_max - 1])) 

class FeaturesException(Exception):
    pass