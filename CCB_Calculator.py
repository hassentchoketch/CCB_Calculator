import numpy as np
import itertools
import statsmodels.api as sm

class CCB_Calculator:
    def __init__(self, df,basket,benchmark_date):
        self.df = df
        self.basket = basket
        self.benchmark_date = benchmark_date

    def _divide_dicts(self, dict1, dict2):
        result_dict = {}
        for key in dict1:
            if key in dict2:
                # Perform division if the key is present in both dictionaries
                result_dict[key] = dict1[key] / dict2[key]
        return result_dict

    def _multiply_dicts(self, dict1, dict2):
        result_dict = {}
        for key in dict1:
            if key in dict2:
                # Perform multiplication if the key is present in both dictionaries
                result_dict[key] = dict1[key] * dict2[key]
        return result_dict

    def _multiply_dict_by_float(self, dictionary, multiplier):
        result_dict = {}
        for key, value in dictionary.items():
            if isinstance(value, float):
                result_dict[key] = value * multiplier
            else:
                result_dict[key] = value
        return result_dict

    def frenkel_simulation(self, weights, fix=None):
        weights = {'USD': weights[0], 'EUR': weights[1], 'OIL': weights[2]}
        value_us_bd = {
            'USD': self.df.loc[self.benchmark_date, 'USD'],
            'EUR': self.df.loc[self.benchmark_date, 'EUR'],
            'OIL': self.df.loc[self.benchmark_date, 'OIL']
        }
        df_to_dict = self.df[self.basket].to_dict(orient='records')

        if fix:
            relative_weights = self._divide_dicts(weights, value_us_bd)
            exch = self.df['EXC'].loc[self.benchmark_date]
            absolute_coefficient = self._multiply_dict_by_float(relative_weights, exch)
            CCBD = []
            for row_dict in df_to_dict:
                exc = self._multiply_dicts(absolute_coefficient, row_dict)
                exc = sum(exc.values())
                CCBD.append(exc)
        else:
            relative_weights = [self._divide_dicts(weights, row_dict) for row_dict in df_to_dict]
            exch = self.df['EXC']
            CCBD = [sum(self._multiply_dict_by_float(j, i).values()) for j, i in zip(relative_weights, exch)]

        return CCBD

    def get_combinations(self):
        usd_weights, eur_weights, oil_weights = [np.arange(0, 1.1, 0.1)] * 3
        combinations_list = []
        for combination in itertools.product(usd_weights, eur_weights, oil_weights):
            if sum(combination) == 1:
                combinations_list.append(combination)
        return combinations_list

    def get_optimal_weights(self, combinations_list, fix=None):
        df_ = self.df.copy()
        R_2 = float('-inf')
        CCBD = None
        optimal_weights = None
        for weights in combinations_list:
            ccbd = self.frenkel_simulation(list(weights), fix=fix)
            if fix:
                df_['UNDERVALUATION_f'] = df_['EXC'] - ccbd
                X = df_['UNDERVALUATION_f']
            else:
                df_['UNDERVALUATION_v'] = df_['EXC'] - ccbd
                X = df_['UNDERVALUATION_v']

            y = df_['INF']

            X = sm.add_constant(X)
            model = sm.OLS(y, X)
            results = model.fit()
            r_squared = results.rsquared
            if r_squared > R_2:
                R_2 = r_squared
                CCBD = ccbd
                if fix:
                    self.df['UNDERVALUATION_f'] = self.df['EXC'] - CCBD
                else:
                    self.df['UNDERVALUATION_v'] = self.df['EXC'] - CCBD
                optimal_weights = weights

        return optimal_weights, CCBD

    def special_cases(self, combinations_list, fix):
        ccbd_oil = None
        ccbd_eur = None
        ccbd_usd = None
        for weights in combinations_list:
            if weights[2] == 1:
                ccbd_oil = self.frenkel_simulation( list(weights), self.benchmark_date, fix=fix)
            elif weights[1] == 1:
                ccbd_eur = self.frenkel_simulation( list(weights), self.benchmark_date, fix=fix)
            elif weights[0] == 1:
                ccbd_usd = self.frenkel_simulation( list(weights), self.benchmark_date, fix=fix)
        return ccbd_oil, ccbd_eur, ccbd_usd
