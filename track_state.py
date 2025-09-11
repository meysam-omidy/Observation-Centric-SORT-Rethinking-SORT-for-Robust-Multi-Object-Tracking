class Meta(type):
    def __repr__(cls):
        return repr(cls.num)

    @property
    def name(cls):
        return cls.__name__.split('State')[1]
    
    @property
    def num(cls):
        return TrackState.__subclasses__().index(cls)

class TrackState(metaclass=Meta): pass
class StateUnconfirmed(TrackState): pass
class StateTracking(TrackState): pass
class StateLost(TrackState): pass
class StateDeleted(TrackState): pass