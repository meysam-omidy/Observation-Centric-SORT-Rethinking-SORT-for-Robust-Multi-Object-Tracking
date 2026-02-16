import numpy as np
import textwrap
from track_state import StateUnconfirmed, StateTracking, StateLost, StateDeleted, TrackState
from utils import BBOX, get_dict_item, get_dict_key, batch_speed_direction
from pydantic import BaseModel
from typing import Union, Literal

class TrackConfig(BaseModel):
    max_age : int = 30
    update_window_start : int = 3
    update_window_end : int = 6
    delta_t : int = 3
    image_width : int = 1920
    image_height : int = 1080
    reupdate_type : Literal['constant', 'relative', None] = None
    reupdate_constant_weight : float = 1


class TrackHistoryItem:
    def __init__(self, bbox : BBOX, score: float, type : str = None):
        self.bbox = bbox
        self.score = float(score)
        self.type = type

    def __repr__(self):
        if self.type is not None:
            return repr((self.bbox, self.score, self.type))
        else:
            return repr((self.bbox, self.score))
        

class TrackHistory:
    def __init__(self):
        self.update : dict[int, TrackHistoryItem] = {}
        self.predict : dict[int, TrackHistoryItem] = {}
        self.state : dict[int, str] = {}

    def __getitem__(self, key):
        return (
            self.update.get(key) if key in self.update else None,
            self.predict.get(key) if key in self.predict else None,
            self.state.get(key) if key in self.state else None,
        )

    def __repr__(self):
        return repr({
            'update': self.update,
            'predict': self.predict,
            'state': self.state,
        })


class Track:
    def __init__(self, 
                 bbox : Union[list, np.ndarray], 
                 score : float, 
                 id : int, 
                 frame_number : int, 
                 config : dict = {}, 
                 state : TrackState = StateUnconfirmed):
        self.config = TrackConfig.model_validate(config)
        self.state = state
        self.last_state = None
        self.age = 0
        self.frame_count = 0
        self.entered_frame = frame_number
        self.exited_frame = -1
        self.history = TrackHistory()
        self.history.update[self.current_frame] = TrackHistoryItem(
            BBOX.from_tlbr(bbox),
            score
        )
        self.history.predict[self.current_frame] = TrackHistoryItem(
            BBOX.from_tlbr(bbox),
            score
        )
        self.history.state[self.current_frame] = self.state.name
        self.current_frame_update = None
        self.logs = {
            'max_time_lost': 0
        }
        self.id = id

    def __str__(self):
        return self.clean_format
    
    def __repr__(self):
        return repr(self.compressed_format)

    @property
    def bbox(self) -> BBOX:
        return get_dict_item(self.history.predict, -1).bbox
            
    @property
    def k_last_updates(self) -> list:
        k_last = []
        for i in range(self.current_frame - self.config.update_window_end, self.current_frame):
            if i in self.history.update:
                k_last.append(self.history.update[i])
            elif i in self.history.predict:
                k_last.append(self.history.predict[i])
        return k_last
    
    @property
    def current_frame(self) -> int:
        return self.entered_frame + self.frame_count

    @property
    def mot_format(self) -> str:
        tlwh = get_dict_item(self.history.update, -1).bbox.to_tlwh()
        return (
            f"{self.current_frame},"
            f"{int(self.id)},"
            f"{round(tlwh[0], 1)},"
            f"{round(tlwh[1], 1)},"
            f"{round(tlwh[2], 1)},"
            f"{round(tlwh[3], 1)},"
            f"{round(self.score, 1)},"
            f"-1,-1,-1"
        )

    @property
    def clean_format(self) -> str:
        return textwrap.dedent(f"""
            **************************************************************************************************************
            id         -> {self.id}
            state      -> {self.state.name}
            bbox       -> {get_dict_item(self.history.update, -1).bbox.to_tlwh()}
            age        -> {self.age}
            score      -> {self.score}
            entered    -> {self.entered_frame}
            {f'exited     -> {self.exited_frame}' if self.state == StateDeleted else ''}
            {f'last state -> {self.last_state.name}' if self.last_state else ''}
            """).strip()
    
    @property
    def compressed_format(self) -> str:
        return (
            f"{self.state.name}    "
            f"{self.id}    "
            f"{get_dict_item(self.history.update, -1).bbox.to_tlwh()}    "
            f"{self.age}    "
            f"{self.score}    "
            f"{self.entered_frame}    "
            f"{self.exited_frame}    "
            f"{self.last_state.name if self.last_state else ''}"

        )

    @property
    def score(self) -> float:
        if len(self.history.update) > 0:
            return np.mean([history_item.score for history_item in self.history.update.values()]).item()
        else:
            return 0
    
    @property
    def k_last_observation(self) -> np.ndarray:
        for i in range(self.config.delta_t, 0, -1):
            k = self.current_frame - i - 1
            if k in self.history.update:
                return self.history.update[k].bbox.to_tlbr()
        if len(self.history.update) < 2:
            return np.array([0, 0, 1, 1])
        else:
            return get_dict_item(self.history.update, -2).bbox.to_tlbr()
    
    @property
    def speed_direction(self) -> np.float64:
        if len(self.history.update) < 2:
            return np.float64(0)
        bbox_last = get_dict_item(self.history.update, -1).bbox.to_tlbr()
        # bbox_last = self.bbox.to_tlbr().reshape(1, 4)
        bbox_k_last = self.k_last_observation.reshape(1, 4)
        return batch_speed_direction(bbox_k_last, bbox_last)[0,0]

    @property
    def is_valid(self) -> bool:
        invalid_conditions = [
            self.age > self.config.max_age,
            self.state == StateUnconfirmed and self.age >= 2,
            np.any(self.bbox[2:] <= 0)
        ]   
        if any(invalid_conditions):
            return False
        else:
            return True
        
    def predict(self):
        self.age += 1
        self.frame_count += 1
        if self.state == StateTracking and self.age >= 2:
            self.state = StateLost
        self.history.state[self.current_frame] = self.state.name
            
    def update(self, 
               bbox : Union[list, np.ndarray], 
               score : float):
        require_reupdate = False
        if self.age > 1:
            require_reupdate = True
        self.history.update[self.current_frame] = TrackHistoryItem(
            BBOX.from_tlbr(bbox),
            score
        )
        self.logs['max_time_lost'] = max(self.age, self.logs['max_time_lost'])
        self.age = 0
        if self.state == StateUnconfirmed:
            self.state = StateTracking
        if self.state == StateLost:
            self.state = StateTracking
            self.last_state = None
        if require_reupdate and self.config.reupdate_type is not None:
            self.reupdate()

    def reupdate(self):
        new_frame_index = get_dict_key(self.history.update, -1)
        last_frame_index = get_dict_key(self.history.update, -2)
        boxes = np.linspace(
            get_dict_item(self.history.update, -2).bbox, 
            get_dict_item(self.history.update, -1).bbox, 
            new_frame_index - last_frame_index + 1
        )
        if self.config.reupdate_type == 'constant':
            scores = [self.config.reupdate_constant_weight for _ in range(len(boxes))]
        else:
            scores = np.linspace(
                get_dict_item(self.history.update, -2).score, 
                get_dict_item(self.history.update, -1).score, 
                new_frame_index - last_frame_index + 1
            )
        for i in range(new_frame_index - last_frame_index + 1):
            self.history.update[last_frame_index + i] = TrackHistoryItem(boxes[i], scores[i], 'virtual')





        
