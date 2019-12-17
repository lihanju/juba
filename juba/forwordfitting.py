import numpy as np
import matplotlib.pyplot as plt


class ForwordFitting(object):

    def __init__(self, periodic_factor=2, complexity=10):
        self.pred_step = periodic_factor
        self.complexity = complexity
        self.coef_ak = []
        self.coef_bk = []
        self.coefficient =[]
        self.residual_error =[]
        self.regular_value =[]
        self.base_value =[]
        self.cos_coef = []
        self.sin_coef = []

    def fit(self, sequence):
        self.sequence = sequence
        self.sequence_len = len(sequence)
        self.a0 = np.mean(sequence)
        self.intercept = self.a0
        residual = np.array(sequence) - self.a0
        mse_r1 = np.mean(residual*residual)
        self.coef_ak.append(self.a0)
        self.coef_bk.append(0)
        self.coefficient.append((self.a0,0))
        self.residual_error.append(residual)
        self.regular_value.append(mse_r1+self.a0**2/self.sequence_len)
        self.base_value.append(self.a0**2/self.sequence_len)
        for k in range(1,self.complexity+1):
            cos_k = np.array([np.cos(2*np.pi*k*i/(self.sequence_len+self.pred_step)) for i in range(1,self.sequence_len+1)])
            sin_k = np.array([np.sin(2*np.pi*k*i/(self.sequence_len+self.pred_step)) for i in range(1,self.sequence_len+1)])
            sum_cos_square_k = np.sum(cos_k * cos_k)
            sum_sin_square_k = np.sum(sin_k * sin_k)
            sum_cos_sin_k = np.sum(cos_k*sin_k)
            fenwu_k = sum_cos_square_k*sum_sin_square_k - np.square(sum_cos_sin_k)
            a_k = (np.sum(residual*cos_k)*sum_sin_square_k - sum_cos_sin_k*np.sum(residual*sin_k))/fenwu_k
            b_k = (np.sum(residual*sin_k)*sum_cos_square_k - sum_cos_sin_k*np.sum(residual*cos_k))/fenwu_k
            self.coef_ak.append(a_k)
            self.coef_bk.append(b_k)
            self.cos_coef.append(a_k)
            self.sin_coef.append(b_k)
            self.base_value.append(self.base_value[-1]+(a_k**2+b_k**2)/self.sequence_len)
            beta_k = np.array([a_k*np.cos(2*np.pi*k*i/(self.sequence_len+self.pred_step)) + b_k*np.sin(2*np.pi*k*i/(self.sequence_len+self.pred_step)) for i in range(1,self.sequence_len+1)])
            residual = residual -beta_k
            mse_r_k_plus_one = np.mean(residual*residual)
            regular_value = mse_r_k_plus_one +self.base_value[-1]
            self.regular_value.append(regular_value)
            self.coefficient.append((a_k,b_k))
            self.residual_error.append(residual)
        self.best_complexity = np.argmin(np.array(self.regular_value))

    def interpolate(self, x):
        y=self.a0
        for k in range(1,self.complexity+1):
            y = y + self.coefficient[k][0]*np.cos(2*np.pi*k*x/(self.sequence_len+self.pred_step))+self.coefficient[k][1]*np.sin(2*np.pi*k*x/(self.sequence_len+self.pred_step))
        return y

    def fit_original_data(self):
        return [self.interpolate(i) for i in range(1, self.sequence_len + 1)]

    def predict(self,pred_step=5):
        x = np.linspace(1,self.sequence_len+pred_step,(self.sequence_len+pred_step)*8)
        plt.plot(x,self.interpolate(x),label='predicted value')
        plt.plot(range(1,self.sequence_len+1),self.sequence,label='original value')
        plt.grid()
        plt.legend()
        plt.show()
        return [self.interpolate(i) for i in range(self.sequence_len + 1, self.sequence_len + 1 + pred_step)]

class FeatureExtraction(ForwordFitting):

    def __init__(self,periodic_factor=50,complexity=10):
        super(FeatureExtraction,self).__init__(periodic_factor,complexity)

    def feature_extraction(self, sequence):
        sequence = np.array(sequence)
        self.fit(sequence)
        return self.coef_ak+self.coef_bk[1:]

    def metric_feature_extraction(self,sequence):
        sequence = np.array(sequence)
        self.fit(sequence)
        return np.array(self.coef_ak)**2+np.array(self.coef_bk)**2

    def plot_coefficient(self):
        plt.plot(self.coef_ak,self.coef_bk)
        plt.grid()
        plt.show()

if __name__ == '__main__':
    data = [47.06, 36.17, 32.31, 44.61, 46.1, 48.07, 52.13, 56.05, 59.34, 50.88, 48.76, 45.21, 45.02, 29.76, 39.81,
            46.19, 47.09, 53.56, 54.32, 58.16, 61.04, 53.85, 50.23, 42.14, 51.01, 44.25, 26.21, 46.67, 47.67, 58.59,
            63.33, 62.87, 57.99, 53.09, 50.46, 50.6, 39.8, 34.12, 51.2, 50.95, 58.2, 63.31, 70.87, 68.62, 64.08, 56.73,
            51.17, 51.92, 50.66]
    F = ForwordFitting(periodic_factor=len(data), complexity=30)
    F.fit(sequence=data)
    print(F.predict(pred_step=5))
    print(F.fit_original_data())
    print(F.best_complexity)
    print(F.intercept)
    print(F.cos_coef)
    print(F.sin_coef)
    # fe=FeatureExtraction(periodic_factor=len(sequece),complexity=22)
    # fe.fit(sequence=sequece)
    # fe.plot_coefficient()
    # fe.predict(pred_step=1)
    # print(fe.best_complexity)