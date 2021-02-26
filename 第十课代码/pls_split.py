import numpy as np
import scipy.io as sio
from sklearn.cross_validation import train_test_split


class pls_split():
    def __init__(self, x_cal, y_cal):
        self.x_cal = x_cal
        self.y_cal = y_cal
    def split(self, intervals):  # 
        n, m = np.shape(self.x_cal)
#         
        num = m / intervals  #  
        mod = m % intervals  #  
        before = num + 1
        split_list = []
        intervals_list = []
        for i in range(mod):
            self.intervals = self.x_cal[:, i * before:(i + 1) * before]
            split_list.append(self.intervals)
            before_intervals = (i * before, (i + 1) * before)
            intervals_list.append(before_intervals)
        before_num = mod * before

        behind = intervals - mod

        for i in range(behind):
            self.intervals = self.x_cal[:, before_num + i * num:before_num + (i + 1) * num]
            split_list.append(self.intervals)
            behind_intervals = (before_num + i * num, before_num + (i + 1) * num)
            intervals_list.append(behind_intervals)
            

        return split_list, intervals_list





