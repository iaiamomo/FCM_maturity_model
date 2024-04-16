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

        plt.figure()
        axes = plt.axes()

        for i in self.mfs:
            axes.plot(self.range_values, self.mfs[i], linewidth=0.5, label=str(i))
            axes.fill_between(self.range_values, self.mfs[i], alpha=0.5)

        axes.legend(loc='lower left')

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        
        values = []
        for key in self.linguistic_terms.keys():
            values.append(self.linguistic_terms[key][1])
        if 1.0 not in values:
            values.append(1.0)
        plt.xticks(values)

        plt.xlabel('Value')
        plt.ylabel('Membership')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.show(block=False)

    def get_linguisitic_term(self, value):
        assert self.mfs is not None

        vals = {}
        for term in self.linguistic_terms.keys():
            res = fuzz.interp_membership(self.range_values, self.mfs[term], value)
            if res > 0:
                vals[res] = term
        if len(vals) > 0:
            max_val = max(vals)
            return vals[max_val]

        return None

    def get_value(self, term):
        assert self.mfs is not None

        if term not in self.linguistic_terms.keys():
            return 0

        return self.linguistic_terms[term][1]


# activation level linguistic terms
def define_al_fuzzy():
    range_values = np.arange(0, 1.001, 0.001)
    linguistic_terms = {
                        'NA':   [0, 0, 0.001],
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
    range_values = np.arange(0, 1.001, 0.05)
    linguistic_terms = {
                        'NA':   [0, 0, 0.25],
                        'L':   [0, 0.25, 0.5],
                        'M':   [0.25, 0.5, 0.75],
                        'H':   [0.5, 0.75, 1.],
                        'VH':   [0.75, 1, 1]
                        }
    flt = Fuzzy_Linguistic_Terms(range_values, linguistic_terms)
    return flt


if __name__ == "__main__":
    flt = define_al_fuzzy()
    # extract values
    res = [flt.get_value(x) for x in flt.linguistic_terms.keys()][1:]
    print(res)
    res = flt.get_linguisitic_term(0.49)
    print(res)
