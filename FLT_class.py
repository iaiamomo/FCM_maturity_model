import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

class Fuzzy_Linguistic_Terms:
    
    def __init__(self, range_values, linguistic_terms, type_mf='triangular'):
        self.range_values : np.array = range_values
        self.linguistic_terms : dict = linguistic_terms
        if type_mf == 'triangular':
            self.mfs = self.set_triangular_membership()
        elif type_mf == 'trapezoidal':
            self.mfs = self.set_trapezoidal_membership()

    def set_triangular_membership(self):
        assert self.range_values is not None
        assert self.linguistic_terms is not None

        mfs = {}
        for term in self.linguistic_terms.keys():
            mfs[term] = fuzz.trimf(x = self.range_values, abc=self.linguistic_terms[term])

        return mfs
    
    def set_trapezoidal_membership(self):
        assert self.range_values is not None
        assert self.linguistic_terms is not None

        mfs = {}
        for term in self.linguistic_terms.keys():
            mfs[term] = fuzz.trapmf(x = self.range_values, abcd=self.linguistic_terms[term])
        
        return mfs

    def plot_triangle(self):
        assert self.mfs is not None

        #plt.figure(figsize = (10, 5))
        plt.figure()
        axes = plt.axes()

        for i in self.mfs:
            axes.plot(self.range_values, self.mfs[i], linewidth=0.5, label=str(i))
            axes.fill_between(self.range_values, self.mfs[i], alpha=0.5)

        #axes.legend(bbox_to_anchor=(0.95, 0.6))
        axes.legend(loc='lower left')

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        #axes.get_xaxis().tick_bottom()
        #axes.get_yaxis().tick_left()
        #set x tick value
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

        for term in self.linguistic_terms.keys():
            if fuzz.interp_membership(self.range_values, self.mfs[term], value) > 0:
                return term

        return None

    def get_value(self, term):
        assert self.mfs is not None

        if term not in self.linguistic_terms.keys():
            return 0

        return self.linguistic_terms[term][1]


# activation level linguistic terms
def define_al_fuzzy():
    range_values = np.arange(0, 1.001, 0.001)
    # 0.1, 0.3, 0.5, 0.7, 0.9
    linguistic_terms = {
                        'L':   [0, 0.25, 0.5],
                        'M':   [0.25, 0.5, 0.75],
                        'H':   [0.5, 0.75, 1.]
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
    flt = define_wm_fuzzy()
    flt.plot_triangle()
    plt.show(block=False)

    flt = define_al_fuzzy()
    flt.plot_triangle()
    plt.show()