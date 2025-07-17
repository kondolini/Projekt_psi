from typing import Optional
from .race_participation import RaceParticipation


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
