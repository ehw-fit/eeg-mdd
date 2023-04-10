#%pip install svgutils

#%%%
import svgutils.transform as sg
import os
# %%
import matplotlib.pyplot as plt
from matplotlib import colors

class EEGschema:
    def __init__(self, cmap = "viridis"):
        self.f = sg.fromfile(os.path.join(os.path.dirname(__file__), "../fig/schema.svg"))
        self.cmap = plt.get_cmap(cmap)        

    def set_channel_color(self, channel, c):
        if type(c) is float:
            c = self.cmap(c)
        c = colors.to_hex(c, keep_alpha=False)

        try:
            circle = self.f.find_id(f"sig-{channel}")[0].root
        except IndexError:
            raise KeyError(f"Unknown channel {channel}")
        settings = dict(tuple(x.split(":")) for x in circle.attrib["style"].split(";"))
        settings["fill"] = c

        circle.attrib["style"] = ";".join([":".join(x) for x in settings.items()])
        #= 'fill:#ff000000;stroke:#000000;stroke-width:0.499999'
        
        return self

    def savefig(self, fname, encoding=None):
        self.f.save(fname, encoding=encoding)

    def _repr_svg_(self):
        return self.f.to_str().decode()

if __name__ == "__main__":

    s = EEGschema()
    s.set_channel_color("c3", 0.4)
    s.set_channel_color("f3", "red")
    s.savefig("../test.svg")