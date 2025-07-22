from typing import Optional
from models.race_participation import RaceParticipation


class Track:
    def __init__(
        self,
        name: str,
        surface_type: Optional[str] = None,
        geometry: Optional[str] = None,
        notes: Optional[str] = None,
    ):
        self.name = name
        self.surface_type = surface_type
        self.geometry = geometry
        self.notes = notes

    def __repr__(self):
        return f"Track(name={self.name})"
    
    def print_info(self):
        print({
            "name": self.name,
            "surface_type": self.surface_type,
            "geometry": self.geometry,
            "notes": self.notes,
        }) 

    @classmethod
    def from_race_participations(cls, participations: list[RaceParticipation]) -> "Track":
        if not participations:
            raise ValueError("No participations provided to construct Track")

        first = participations[0]
        return cls(
            name=first.track_name,
            surface_type=None,  # Extend later if available in source
            geometry=None,      # Extend later if available in source
            notes=None          # Optional notes field
        )

def load_all_tracks(tracks_dir: str = "data/tracks") -> dict:
    """Load all track objects from pickle files in the tracks directory"""
    import os
    import pickle
    
    tracks = {}
    if not os.path.exists(tracks_dir):
        print(f"Warning: Tracks directory {tracks_dir} does not exist")
        return tracks
    
    for fname in os.listdir(tracks_dir):
        if fname.endswith(".pkl"):
            try:
                with open(os.path.join(tracks_dir, fname), "rb") as f:
                    track = pickle.load(f)
                    tracks[track.name] = track
            except Exception as e:
                print(f"Error loading track from {fname}: {e}")
                continue
    
    return tracks
