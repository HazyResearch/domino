from dataclasses import dataclass


class AbstractSliceBuilder:
    @dataclass
    class Config:
        pass

    def build_correlation_slices(self):
        raise NotImplementedError

    def build_rare_slices(self):
        raise NotImplementedError

    def build_noisy_label_slices(self):
        raise NotImplementedError

    def buid_noisy_feature_slices(self):
        raise NotImplementedError
