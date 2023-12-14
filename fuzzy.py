import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

class Fuzzy_Linguistic_Terms:
    
    def __init__(self, range_values, linguistic_terms):
        self.range_values : np.array = range_values
        self.linguistic_terms : dict = linguistic_terms
        self.mfs = self.set_triangular_membership()

    def set_triangular_membership(self):
        assert self.range_values is not None
        assert self.linguistic_terms is not None

        mfs = {}
        for term in self.linguistic_terms.keys():
            mfs[term] = fuzz.trimf(x = self.range_values, abc=self.linguistic_terms[term])

        return mfs

    def plot_triangle(self):
        assert self.mfs is not None

        plt.figure(figsize = (10, 5))
        axes = plt.axes()

        for i in self.mfs:
            axes.plot(self.range_values, self.mfs[i], linewidth=0.4, label=str(i))
            axes.fill_between(self.range_values, self.mfs[i], alpha=0.5)

        axes.legend(bbox_to_anchor=(0.95, 0.6))

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        axes.get_xaxis().tick_bottom()
        axes.get_yaxis().tick_left()
        plt.tight_layout()
        plt.show(block=False)

    def get_linguisitic_term(self, value):
        assert self.mfs is not None

        for term in self.linguistic_terms.keys():
            if fuzz.interp_membership(self.range_values, self.mfs[term], value) > 0:
                return term

        return None


# activation level linguistic terms
def define_al_fuzzy():
    range_values = np.arange(0, 1.1, 0.1)
    # 0.1, 0.3, 0.5, 0.7, 0.9
    linguistic_terms = {
                        'VL':   [-0.1, 0.1, 0.3],
                        'L':    [0.1, 0.3, 0.5],
                        'M':    [0.3, 0.5, 0.7],
                        'H':    [0.5, 0.7, 0.9],
                        'VH':   [0.7, 0.9, 1.1]
                        }
    flt = Fuzzy_Linguistic_Terms(range_values, linguistic_terms)
    return flt

# weight matrix linguistic terms
def define_wm_fuzzy():
    range_values = np.arange(-1.1, 1.1, 0.1)
    # -0.8, -0.5, -0.2, 0, 0.2, 0.5, 0.8
    linguistic_terms = {
                        'NH':   [-1.1, -0.8, -0.5],
                        'NM':   [-0.8, -0.5, -0.2],
                        'NL':   [-0.5, -0.2, 0],
                        'NA':   [-0.2, 0, 0.2],
                        'PL':   [0, 0.2, 0.5],
                        'PM':   [0.2, 0.5, 0.8],
                        'PH':   [0.5, 0.8, 1.1]
                        }
    flt = Fuzzy_Linguistic_Terms(range_values, linguistic_terms)
    return flt


if __name__ == "__main__":
    range_values = np.arange(0, 1.1, 0.1)
    linguistic_terms = {
                        'NA': [0, 0, 0.1],
                        'L': [0, 0.2, 0.4],
                        'M': [0.3, 0.5, 0.7],
                        'H': [0.6, 0.8, 1]
                        }
    
    flt = Fuzzy_Linguistic_Terms(range_values, linguistic_terms)

    flt.plot_triangle()