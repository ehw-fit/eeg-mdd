# %%
from __future__ import annotations
from .gaops import GAops, FeaturesException
from .eeg_schema import EEGschema
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re


class ChromosomeBase:
    def __init__(self, ops: GAops):
        self.ops = ops
        self.chrom = []
        self._features = 0
        self.genes_features = None
        self.parameters = None

    def _invalidate(self):
        self._features = None
        self.parameters = None
        self.genes_features = None

    def _assert_length(self):
        assert len(
            self.chrom) < self.ops.max_length, f"Very long chromosome ({len(self.chrom)})"

    def invalidate_features(function):
        def wrapper(self, *args, **kwargs):
            self._invalidate()
            ret = function(self, *args, **kwargs)
            self._assert_length()
            return ret

        return wrapper

    @property
    def features(self) -> int:
        if self._features is None:
            raise FeaturesException(
                "Unknown number of features, run execute first")

        return self._features

    def _random_gene(self):
        chan = random.randint(0, self.ops.channels_max - 1)
        freq_start = random.randint(0, self.ops.freqs_max - 1)
        freq_len = random.randint(1, 20)
        fun = random.randint(0, self.ops.fun_max - 1)

        return (chan, freq_start, freq_len, fun)

    @invalidate_features
    def random(self, length=None) -> Chromosome:

        if not length:
            length = random.randint(1, self.ops.max_length - 1)

        assert length < self.ops.max_length, f"Random lenght is larger than maximal allowed {self.ops.max_length}"

        chrom = []
        for _ in range(length):
            chrom.append(self._random_gene())
        self.chrom = chrom
        return self

    @invalidate_features
    def crossover(self, chrom_a: Chromosome, chrom_b: Chromosome) -> Chromosome:
        chrom = []
        for d in chrom_a.chrom + chrom_b.chrom:
            if random.choice([True, False]):
                chrom.append(d)
        self.chrom = chrom

        while len(self.chrom) >= self.ops.max_length:
            self._mutate_delete()

        return self

    # Mutation functions
    def mutate(self, mutation_count=1, inplace=False) -> Chromosome:
        if inplace:
            cmut = self
        else:
            cmut = self.__class__(self.ops)
            cmut.chrom = self.chrom.copy()

        for _ in range(mutation_count):
            if not self.chrom:
                fn = cmut._mutate_add
            if len(self.chrom) == 1:
                fn = random.choice([
                    cmut._mutate_add,
                    cmut._mutate_modify
                ])
            else:
                fn = random.choice([
                    cmut._mutate_delete,
                    cmut._mutate_add,
                    cmut._mutate_modify
                ])
            fn()

        cmut._invalidate()
        cmut._assert_length()
        return cmut

    def _mutate_delete(self) -> Chromosome:
        if len(self.chrom) <= 1:
            return self
        i = random.randint(0, len(self.chrom) - 1)

        len1 = len(self.chrom)
        self.chrom.pop(i)
        len2 = len(self.chrom)

        assert len1 == len2 + 1
        return self

    def _mutate_modify(self) -> Chromosome:
        if not len(self.chrom):
            return self

        i = random.randint(0, len(self.chrom) - 1)

        new_gene = self._random_gene()

        j = random.randint(0, 3)
        old_gene = list(self.chrom[i])
        old_gene[j] = new_gene[j]
        self.chrom[i] = tuple(old_gene)
        return self

    def _mutate_add(self):
        self.chrom.append(self._random_gene())
        while len(self.chrom) >= self.ops.max_length:
            self._mutate_delete()
        return self

    def _gene_to_str(self, gene):
        ch, fs, fl, fun = gene
        fstart = self.ops.freqs[fs]
        fend = self.ops.freqs[fs + self.freq_lenght(fs, fl, fun)]
        cs = f"{self.ops.channels[ch]}, {fstart}, {fend}, {self.fn2str(fun)}"

        return cs

    def __str__(self):
        s = []
        for gene in self.chrom:
            cs = self._gene_to_str(gene)
            s.append(f"({cs})")

        return "".join(s)

    @invalidate_features
    def from_str(self, data: str) -> ChromosomeBase:

        self.chrom = []
        for l in re.findall(r"\(([^,]+?)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s(\S+)\s*\)", data):
            ch, fs, fe, fun = l
            chid = np.where(self.ops.channels == ch)[0]
            if len(chid) != 1:
                raise KeyError("Channel \"{ch}\" was not found in the list {self.ops.channels}")
            chid = chid[0]
            fs = np.where(self.ops.freqs == float(fs))[0][0]
            fe = np.where(self.ops.freqs == float(fe))[0][0]

            fl = fe - fs
            fun = self.str2fn(fun)
            self.chrom.append((chid, fs, fl, fun))
        return self

    _fndict = {0: "CP", 1: "DS2", 2: "DS4", 3: "DS8", 4: "AGG"}

    def fn2str(self, fn: int) -> str:
        return self._fndict[fn]

    def str2fn(self, fn: str) -> int:
        for k, v in self._fndict.items():
            if v == fn:
                return k
        raise KeyError(f"Unknown function  {fn}")

    def freq_lenght(self, fs, fl, fun) -> int:
        fl = min(fl, self.ops.freqs_max - fs - 1)

        if fun == 1:
            fl = fl - (fl % 2)
        if fun == 2:
            fl = fl - (fl % 4)
        if fun == 3:
            fl = fl - (fl % 8)
        return fl

    def execute(self, features: np.array) -> np.array:
        if features.shape[2] < self.ops.freqs_max:
            self._features = 0
            raise FeaturesException(
                f"Expected size of features freqs {self.ops.freqs_max}, got {features.shape[2]}")

        assert features.shape[
            1] == self.ops.channels_max, f"Expected size of features channels {self.ops.channels_max}, got {features.shape[1]}"

        ret = []
        genes_features = []
        for ch, fs, fl, fun in self.chrom:
            gene_str = self._gene_to_str((ch, fs, fl, fun))

            fstart = fs
            fe = fs + self.freq_lenght(fs, fl, fun)

            finput = features[:, ch, fs:fe]

            foutput = finput

            if fun == 0:
                foutput = finput
            elif fun == 1:  # downsample 2
                foutput = (finput[:, ::2] + finput[:, 1::2]) / 2.0
            elif fun == 2:  # downsample 4
                foutput = (finput[:, ::4] + finput[:, 1::4] +
                           finput[:, 2::4] + finput[:, 3::4]) / 4.0
            elif fun == 3:  # downsample 8
                foutput = (finput[:, ::8] + finput[:, 1::8] + finput[:, 2::8] + finput[:, 3::8] +
                           finput[:, 4::8] + finput[:, 5::8] + finput[:, 6::8] + finput[:, 7::8]) / 8.0
            elif fun == 4:  # aggregate
                if finput.shape[0] * finput.shape[1] == 0:
                    continue
                try:
                    foutput = np.concatenate([r.reshape(-1, 1) for r in [finput.max(axis=1), finput.min(
                        axis=1), np.mean(finput, axis=1), finput.max(axis=1) - finput.min(axis=1)]], axis=1)
                except Exception as e:
                    print(e)
                    print(finput.shape, foutput.shape)
                    raise e
                assert foutput.shape[1] == 4, "unkn {}".format(foutput.shape)
            else:
                raise ValueError(f"Unknown function number {fun}")

            #print("fun", self.fn2str(fun), "in", finput.shape, " out ", foutput.shape)
            genes_features += [gene_str] * foutput.shape[1]
            ret.append(foutput)

        self.genes_features = genes_features
        if not ret:
            self._features = 0
            raise FeaturesException(
                "No features were extracted for this chromosome")

        ret = np.concatenate(ret, axis=1)
        self._features = ret.shape[1]

        assert self._features is not None

        if self._features == 0:
            raise FeaturesException(
                "No features were extracted for this chromosome")
        return ret

    def vizualize(self, ax=None, channel_formatter = None):
        if not ax:
            ax = plt.gca()

        clr = ["tab:green", "tab:red", "tab:orange", "tab:purple", "tab:blue"]

        for ch, fs, fl, fun in self.chrom:

            fstart = self.ops.freqs[fs]
            fe = self.ops.freqs[fs + self.freq_lenght(fs, fl, fun)]

            rect = patches.Rectangle(
                (fstart, ch), fe - fstart, 1, alpha=0.5, facecolor=clr[fun])
            ax.add_patch(rect)

        if not channel_formatter:
            channel_formatter = lambda x: x
        ax.set_yticks(np.arange(self.ops.channels_max) +
                      0.5, map(channel_formatter, self.ops.channels))

        handles = []
        for i in range(self.ops.fun_max):
            handles.append(
                patches.Patch(color=clr[i], label=self.fn2str(i), alpha=0.5)
            )

        self._legend = handles, ax.get_legend_handles_labels()[1]

        ax.legend(handles=handles, loc="best", ncol=1)

        ax.set(
            xlim=(0, self.ops.freqs.max()),
            ylim=(0, self.ops.channels_max),
            xlabel="Frequency [Hz]",
            ylabel="EEG Channel"
        )

        return ax

    def plot_head(self, used_color = "tab:red") -> EEGschema:
        sch = EEGschema()

        for ch, fs, fl, fun in self.chrom:
            chan = self.ops.channels[ch].replace("EEG ", "").replace("-LE", "").lower()
            print(chan)
            sch.set_channel_color(chan, used_color)

        return sch

    def get_heat(self):
        heat = np.full((self.ops.fun_max, self.ops.channels_max,
                       self.ops.freqs_max), False, dtype="b")

        for ch, fs, fl, fun in self.chrom:

            fstart = fs
            fe = fs + self.freq_lenght(fs, fl, fun)

            heat[fun, ch, fstart:fe] = True

        return heat

    @property
    def genes_count(self) -> int:
        return len(self.chrom)


class ChromosomeChannels(ChromosomeBase):
    #_ranges = [0, 4, 8, 12, 20, 30, 50, 75, 100]
    _ranges = [0, 4, 8, 12, 20, 30, 50]

    def _random_gene(self):
        chan = random.randint(0, self.ops.channels_max - 1)
        rs = np.random.randint(0, len(self._ranges) - 1)

        frange = np.where((self.ops.freqs >= self._ranges[rs]) & (
            self.ops.freqs < self._ranges[rs+1]))[0]
        freq_start = frange[0]
        freq_len = len(frange)
        fun = random.randint(0, self.ops.fun_max - 1)

        return (chan, freq_start, freq_len, fun)

    def _mutate_modify(self) -> Chromosome:
        if not(len(self.chrom)):
            return self
        i = random.randint(0, len(self.chrom) - 1)

        new_gene = self._random_gene()

        old_gene = list(self.chrom[i])
        j = random.randint(0, 2)
        if j == 0:
            old_gene[0] = new_gene[0]
        if j == 1:
            old_gene[1] = new_gene[1]
            old_gene[2] = new_gene[2]
        if j == 2:
            old_gene[3] = new_gene[3]

        self.chrom[i] = tuple(old_gene)
        return self


class ChromosomeWindow:
    def __init__(self, ops: GAops, t: type):
        self.chrom = t(ops)
        self.ops = ops
        self.chrom_type = t
        self._window_size = None
        self.window_range = ops.window_range

    @property
    def window_size(self):
        return self._window_size

    @window_size.setter
    def window_size(self, val):
        assert val and self.window_range[0] <= val <= self.window_range[
            1], f"Window size {self.window_size} not in valid range"
        self._window_size = val
        return val

    def _validate_range(self):
        assert self.window_range[0] <= self.window_size <= self.window_range[
            1], f"Window size {self.window_size} not in valid range"
        assert self.window_size > 0

    @property
    def features(self) -> int | None:
        return self.chrom._features

    @property
    def parameters(self):
        return self.chrom.parameters

    @parameters.setter
    def parameters(self, val):
        self.chrom.parameters = val

    def random(self, length=None) -> ChromosomeWindow:
        self.window_size = random.randint(*self.window_range)
        self.chrom.random(length)
        self._validate_range()
        return self

    def crossover(self, chrom_a: ChromosomeWindow, chrom_b: ChromosomeWindow) -> ChromosomeWindow:
        assert chrom_a.window_size
        assert chrom_b.window_size
        self.chrom.crossover(chrom_a.chrom, chrom_b.chrom)

        if random.choice([True, False]):
            self.window_size = chrom_a.window_size
        else:
            self.window_size = chrom_b.window_size

        assert self.window_size
        self._validate_range()
        return self

    # Mutation functions
    def mutate(self, mutation_count=1, inplace=False) -> Chromosome:
        if inplace:
            cmut = self
        else:
            cmut = self.__class__(self.ops, self.chrom_type)
            cmut.chrom.chrom = self.chrom.chrom.copy()
            cmut.window_size = self.window_size

        for _ in range(mutation_count):
            j = random.randint(0, self.chrom.genes_count)
            if j == 0:
                cmut.window_size = random.randint(*self.window_range)
                self._validate_range()
            else:
                cmut.chrom.mutate(1, inplace=True)
        self._validate_range()
        return cmut

    def __str__(self):
        return f"{{{self.window_size}}}{self.chrom}"

    def from_str(self, data: str) -> ChromosomeBase:
        g = re.match(r"\{(\d+)\}(.*)", data)
        ws, ch = g.groups()
        self.chrom.from_str(ch)
        self.window_size = int(ws)
        return self

    def execute(self, features: np.array) -> np.array:
        self._validate_range()
        return self.chrom.execute(features)

    def vizualize(self, ax=None):
        return self.chrom.vizualize(ax)

    def get_heat(self):
        return self.chrom.get_heat()

# %%
