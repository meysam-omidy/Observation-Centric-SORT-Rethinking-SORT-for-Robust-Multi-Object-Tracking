from filterpy.kalman import KalmanFilter as KalmanFilterBase
from utils import tlbr_to_z, get_dict_item, z_to_tlbr, batch_speed_direction, count_time
from copy import deepcopy
import numpy as np

class KalmanFilter(KalmanFilterBase):
    def __init__(self, dim_x, dim_z, z):
        super().__init__(dim_x, dim_z)
        self.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
        self.R[2:,2:] *= 10
        self.P[4:,4:] *= 1000
        self.P *= 10.
        self.Q[-1,-1] *= 0.01
        self.Q[4:,4:] *= 0.01
        self.x[:4] = z
        self.time_since_last_update = 0 
        self.age = 0
        self.dict = None
        self.history = {
            'update': {
                self.age: z
            },
            'predict': {}
        }
        
    def update(self, z, R=None, H=None):
        if self.time_since_last_update > 1:
            self.re_update(z, R, H)
        else:
            super().update(z, R, H)
        self.time_since_last_update = 0
        self.history['update'][self.age] = z
        # self.freeze()
    

    def predict(self, u=None, B=None, F=None, Q=None):
       if self.time_since_last_update == 1:
           self.freeze()
       super().predict(u, B, F, Q)
       self.time_since_last_update += 1
       self.history['predict'][self.age] = self.x
       self.age += 1
    

    def re_update(self, z, R=None, H=None):
        virtual_z = np.linspace(
            self.history['update'][list(self.history['update'].keys())[-1]],
            z,
            self.time_since_last_update + 1
        )
        self.unfreeze()
        for virtual_z_ in virtual_z[1:]:
            super().update(virtual_z_, R, H)
            super().predict()
        self.time_since_last_update = 0
        self.history['update'][self.age] = z

    
    def freeze(self):
        self.__dict__.pop('dict')
        self.dict = deepcopy(self.__dict__)
        self.dict.pop('age')
        self.dict.pop('history')
        self.dict.pop('time_since_last_update')


    def unfreeze(self):
        if self.dict:
            self.dict['age'] = self.age
            self.dict['history'] = self.history
            self.dict['time_since_last_update'] = self.time_since_last_update
            self.__dict__ = deepcopy(self.dict)
            self.dict = None

    
    def k_last_observation(self, delta_t):
        for i in range(delta_t, 0, -1):
            k = self.age - i - 1
            if k in self.history['update']:
                return self.history['update'][k]
        if len(self.history['update']) < 2:
            return np.array([0.5, 0.5, 1, 1])
        else:
            return get_dict_item(self.history['update'], -2)


    def speed_direction(self, delta_t):
        if len(self.history['update']) < 2:
            return np.float64(0)
        box_last = z_to_tlbr(get_dict_item(self.history['update'], -1)).reshape(1, 4)
        box_k = z_to_tlbr(self.k_last_observation(delta_t)).reshape(1, 4)
        return batch_speed_direction(box_k, box_last)[0,0]
 
