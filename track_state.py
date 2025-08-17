class TrackState:
    def __repr__(self):
        return repr(self.num)
    
    @property
    def name(self):
        return self.__class__.__name__.split('State')[1]
    
    @property
    def num(self):
        return TrackState.__subclasses__().index(self.__class__)

class StateUnconfirmed(TrackState): pass
class StateTracking(TrackState): pass
class StateLost(TrackState): pass
class StateDeleted(TrackState): pass

STATE_UNCONFIRMED = StateUnconfirmed()
STATE_TRACKING = StateTracking()
STATE_LOST = StateLost()
STATE_DELETED = StateDeleted()
