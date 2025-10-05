import numpy as np
import textwrap
from track_state import StateUnconfirmed, StateTracking, StateLost, StateDeleted
from utils import z_to_tlwh, z_to_tlbr, z_to_xywh, tlbr_to_z, tlwh_to_z, tlbr_to_tlwh, tlwh_to_tlbr, tlwh_to_xywh, get_r_matrix
from kalman_filter import KalmanFilter, init_kalman_filter
from pydantic import BaseModel

class TrackConfig(BaseModel):
    max_age : int = 30
    delta_t : int = 3


class Track:
    def __init__(self, bbox, score, id, frame_number, config, state=None):
        self.config = TrackConfig.model_validate(config)
        if state == None:
            self.state = StateUnconfirmed
        else:
            self.state = state
        self.last_state = None
        self.kf = init_kalman_filter(tlbr_to_z(bbox), score)
        self.predict_history = []
        self.update_history = [tlbr_to_tlwh(bbox)]
        self.state_history = [self.state]
        self.current_frame_update = None
        self.scores = [float(score)]
        self.age = 0
        self.entered_frame = frame_number
        self.exited_frame = -1
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
        self.kf.predict()
        self.predict_history.append(z_to_tlwh(np.array(self.kf.x)))
        if self.state == StateTracking and self.age >= 2:
            self.state = StateLost
        self.state_history.append(self.state)
            
    def update(self, bbox, score):
        R = get_r_matrix(score)
        # R = np.eye(4)
        # R[2:, 2:] *= 10
        # R *= np.e ** (2 * (1 - score))
        self.kf.update(tlbr_to_z(bbox))
        # self.kf.update(tlbr_to_z(bbox), R=R)
        self.update_history.append(tlbr_to_tlwh(bbox))
        self.scores.append(float(score))
        self.logs['max_time_lost'] = max(self.age, self.logs['max_time_lost'])
        self.age = 0
        if self.state == StateUnconfirmed:
            self.state = StateTracking
        if self.state == StateLost:
            self.state = StateTracking
            self.last_state = None

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
        if len(self.scores) > 0:
            return np.mean(self.scores).item()
            # return float(self.scores[-1])
        else:
            return 0

    @property
    def tlwh(self):
        if self.state == StateTracking:
            return self.update_history[-1]
        else:
            return z_to_tlwh(np.array(self.kf.x))

    @property
    def tlbr(self):
        return z_to_tlbr(np.array(self.kf.x))
    
    @property
    def xywh(self):
        return z_to_xywh(np.array(self.kf.x))
        
    @property
    def xysa(self):
        if self.state == StateTracking:
            return tlwh_to_z(self.update_history[-1]).reshape(-1)[:4]
        else:
            return np.array(self.kf.x).reshape(-1)[:4]
    
    @property
    def speed_direction(self):
        return self.kf.speed_direction(self.config.delta_t)
    
    @property
    def k_last_observation(self):
        return z_to_tlbr(self.kf.k_last_observation(self.config.delta_t))

    @property
    def is_valid(self):
        invalid_conditions = [
            self.age > self.config.max_age,
            self.state == StateUnconfirmed and self.age >= 2 and self.current_frame_update == None,
            np.any(np.isnan(self.kf.x)),
            np.any(self.kf.x[2:4, 0] <= 0)
        ]   
        if any(invalid_conditions):
            return False
        else:
            return True