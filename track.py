import numpy as np
import torch
import textwrap
from track_state import StateUnconfirmed, StateTracking, StateLost, StateDeleted
from utils import tlbr_to_xysa, tlbr_to_tlwh, batch_speed_direction, get_dict_item
from kalman_filter import KalmanFilter, init_kalman_filter
from motion_predictor import model as MODEL, device as DEVICE
from pydantic import BaseModel

class TrackConfig(BaseModel):
    max_age : int = 30
    delta_t : int = 3
    image_width : int = 1920
    image_height : int = 1080


class TrackHistory:
    def __init__(self):
        self.update = {}
        self.predict = {}
        self.state = {}
        self.score = {}

    def __repr__(self):
        return repr({
            'update': self.update,
            'predict': self.predict,
            'state': self.state,
            'score': self.score,
        })


class Track:
    def __init__(self, bbox, score, id, frame_number, config, state=None):
        self.config = TrackConfig.model_validate(config)
        if state == None:
            self.state = StateUnconfirmed
        else:
            self.state = state
        self.last_state = None
        self.age = 0
        self.frame_count = 0
        self.entered_frame = frame_number
        self.exited_frame = -1
        self.history = TrackHistory()
        self.history.update[self.current_frame] = bbox
        self.history.predict[self.current_frame] = bbox
        self.history.state[self.current_frame] = self.state.name
        self.history.score[self.current_frame] = score
        self.current_frame_update = None
        self.logs = {
            'max_time_lost': 0
        }
        self.id = id

    def __str__(self):
        return self.clean_format
    
    def __repr__(self):
        return repr(self.compressed_format)

    def predict(self):
        self.age += 1
        self.frame_count += 1
        if self.state == StateTracking and self.age >= 2:
            self.state = StateLost
        self.history.state[self.current_frame] = self.state.name
            
    def update(self, bbox, score):
        self.history.update[self.current_frame] = bbox
        self.history.score[self.current_frame] = float(score)
        self.logs['max_time_lost'] = max(self.age, self.logs['max_time_lost'])
        self.age = 0
        if self.state == StateUnconfirmed:
            self.state = StateTracking
        if self.state == StateLost:
            self.state = StateTracking
            self.last_state = None
            

    @property
    def k_last_updates(self):
        k_last = []
        for i in range(self.current_frame - 10, self.current_frame):
            if i in self.history.update:
                k_last.append(self.history.update[i])
            elif i in self.history.predict:
                k_last.append(self.history.predict[i])
        return k_last
    
    @property
    def current_frame(self):
        return self.entered_frame + self.frame_count

    @property
    def mot_format(self):
        tlwh = self.tlwh
        return f"{{frame_number}},{int(self.id)},{round(tlwh[0], 1)},{round(tlwh[1], 1)},{round(tlwh[2], 1)},{round(tlwh[3], 1)},{round(self.score, 2)},-1,-1,-1"

    @property
    def clean_format(self):
        return textwrap.dedent(f"""
            **************************************************************************************************************
            id         -> {self.id}
            state      -> {self.state.name}
            bbox       -> {self.tlwh}
            age        -> {self.age}
            score      -> {self.score}
            entered    -> {self.entered_frame}
            {f'exited     -> {self.exited_frame}' if self.state == StateDeleted else ''}
            {f'last state -> {self.last_state.name}' if self.last_state else ''}
            """).strip()
    
    @property
    def compressed_format(self):
        return f"{self.state.name}    {self.id}    {self.tlwh}    {self.age}    {self.score}    {self.entered_frame}    {self.exited_frame}    {self.last_state.name if self.last_state else ''}"

    @property
    def score(self):
        if len(self.history.score) > 0:
            return np.mean(list(self.history.score.values())).item()
            # return float(self.scores[-1])
        else:
            return 0

    @property
    def tlwh(self):
        if self.state == StateTracking:
            return tlbr_to_tlwh(get_dict_item(self.history.update, -1))
        else:
            return tlbr_to_tlwh(get_dict_item(self.history.predict, -1))

    @property
    def tlbr(self):
        return get_dict_item(self.history.predict, -1)
        
    @property
    def xysa(self):
        if self.state == StateTracking:
            return tlbr_to_xysa(get_dict_item(self.history.update, -1))
        else:
            return tlbr_to_xysa(get_dict_item(self.history.predict, -1))
    
    @property
    def k_last_observation(self):
        for i in range(self.config.delta_t, 0, -1):
            k = self.current_frame - i - 1
            if k in self.history.update:
                return self.history.update[k]
        if len(self.history.update) < 2:
            return np.array([0, 0, 1, 1])
        else:
            return get_dict_item(self.history.update, -2)
    
    @property
    def speed_direction(self):
        if len(self.history.update) < 2:
            return np.float64(0)
        bbox_last = get_dict_item(self.history.update, -1).reshape(1, 4)
        bbox_k_last = self.k_last_observation.reshape(1, 4)
        return batch_speed_direction(bbox_k_last, bbox_last)[0,0]

    @property
    def is_valid(self):
        invalid_conditions = [
            self.age > self.config.max_age,
            self.state == StateUnconfirmed and self.age >= 2,
            # self.state == StateUnconfirmed and self.age >= 2 and self.current_frame_update == None,
            # np.any(self.kf.x[2:4, 0] <= 0)
        ]   
        if any(invalid_conditions):
            return False
        else:
            return True