import numpy as np
import textwrap
from track_state import STATE_UNCONFIRMED, STATE_TRACKING, STATE_LOST, STATE_DELETED, TrackState
from utils import z_to_tlwh, z_to_tlbr, z_to_xywh, tlbr_to_z, tlbr_to_tlwh, count_time
from kalman_filter import KalmanFilter

class Track:
    @classmethod
    def init(cls, max_age, min_box_area, max_aspect_ratio, delta_t):
        cls.INSTANCES:list['Track'] = []
        cls.ID_COUNTER = 1
        cls.FRAME_NUMBER = 0
        cls.MAX_AGE = max_age
        cls.MIN_BOX_AREA = min_box_area
        cls.MAX_ASPECT_RATIO = max_aspect_ratio
        cls.DELTA_T = delta_t

    @classmethod
    def get_tracks(cls, included_states:list[TrackState]) -> list['Track']:
        return [track for track in cls.INSTANCES if track.state in included_states]

    @classmethod
    def predict_all(cls) -> None:
        cls.FRAME_NUMBER += 1
        for track in cls.INSTANCES:
            if track.state not in [STATE_DELETED]:
                track.predict()
    
    @property
    def mot_format(self):
        return f"{int(Track.FRAME_NUMBER)},{int(self.id)},{round(self.tlwh[0], 1)},{round(self.tlwh[1], 1)},{round(self.tlwh[2], 1)},{round(self.tlwh[3], 1)},{round(self.score, 2)},-1,-1,-1"

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
            {f'exited     -> {self.exited_frame}' if self.state == STATE_DELETED else ''}
            {f'last state -> {self.last_state.name}' if self.last_state else ''}
            """).strip()
    
    @property
    def compressed_format(self):
        return f"{self.state.name}    {self.id}    {self.tlwh}    {self.age}    {self.score}    {self.entered_frame}    {self.exited_frame}    {self.last_state.name if self.last_state else ''}"

    @property
    def score(self):
        if len(self.scores) > 0:
            return float(self.scores[-1])
        else:
            return 0

    @property
    def tlwh(self):
        return z_to_tlwh(np.array(self.kf.x))

    @property
    def tlbr(self):
        return z_to_tlbr(np.array(self.kf.x))
    
    @property
    def xywh(self):
        return z_to_xywh(np.array(self.kf.x))
    
    @property
    def speed_direction(self):
        return self.kf.speed_direction(Track.DELTA_T)
    
    @property
    def k_last_observation(self):
        return z_to_tlbr(self.kf.k_last_observation(Track.DELTA_T))

    @property
    def valid(self):
        invalid_conditions = [
            self.age > Track.MAX_AGE,
            self.state == STATE_UNCONFIRMED and self.age >= 2,
            self.kf.x[2,0] < Track.MIN_BOX_AREA,
            self.kf.x[3,0] > Track.MAX_ASPECT_RATIO,
            np.any(np.isnan(self.kf.x)) or np.any(self.kf.x[2:4, 0] <= 0)
        ]   
        if any(invalid_conditions):
            return False
        else:
            return True

    def __init__(self, bbox, score, state=None):
        if state == None:
            self.state = STATE_UNCONFIRMED
        else:
            self.state = state
        self.last_state = None
        self.kf = KalmanFilter(dim_x=7, dim_z=4, z=tlbr_to_z(bbox))
        self.predict_history = []
        self.update_history = [self.tlwh]
        self.state_history = [self.state]
        self.scores = [float(score)]
        self.age = 0
        self.entered_frame = Track.FRAME_NUMBER
        self.exited_frame = -1
        self.logs = {
            'max_time_lost': 0
        }
        self.id = Track.ID_COUNTER
        Track.ID_COUNTER += 1
        Track.INSTANCES.append(self)

    def __str__(self):
        return self.clean_format
    
    def __repr__(self):
        return repr(self.compressed_format)

    def predict(self):
        self.age += 1
        self.kf.predict()
        if not self.valid:
            self.last_state = self.state
            self.state = STATE_DELETED
            self.exited_frame = Track.FRAME_NUMBER
            return
        self.predict_history.append(self.tlwh)
        if self.state == STATE_TRACKING and self.age >= 2:
            self.state = STATE_LOST
        self.state_history.append(self.state)
            
    def update(self, bbox, score):
        self.kf.update(tlbr_to_z(bbox))
        self.update_history.append(tlbr_to_tlwh(bbox))
        self.scores.append(float(score))
        self.logs['max_time_lost'] = max(self.age, self.logs['max_time_lost'])
        self.age = 0
        if self.state == STATE_UNCONFIRMED:
            self.state = STATE_TRACKING
        if self.state == STATE_LOST:
            self.state = STATE_TRACKING
            self.last_state = None