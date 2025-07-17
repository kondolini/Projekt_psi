from typing import List, Optional
from datetime import datetime
from .race_participation import RaceParticipation


class Dog:
    def __init__(self, dog_id: str):
        self.id = dog_id
        self.name: Optional[str] = None
        self.birth_date: Optional[datetime] = None
        self.trainer: Optional[str] = None
        self.color: Optional[str] = None
        self.weight: Optional[float] = None
        self.sire: Optional['Dog'] = None
        self.dam: Optional['Dog'] = None

        self.race_participations: List[RaceParticipation] = []

    def set_birth_date(self, birth_date: datetime):
        self.birth_date = birth_date

    def set_color(self, color: str):
        self.color = color

    def set_weight(self, weight: float):
        self.weight = weight

    def add_participations(self, participations: List[RaceParticipation]):
        self.race_participations += participations
        
    def add_participation(self, rp: RaceParticipation):
        self.race_participations.append(rp)
        self.race_participations.sort(key=lambda x: x.race_datetime)

    def get_participations_before(self, dt: datetime) -> List[RaceParticipation]:
        return [rp for rp in self.race_participations if rp.race_datetime < dt]

    def get_participation_by_race_id(self, race_id: str) -> Optional[RaceParticipation]:
        for rp in self.race_participations:
            if rp.race_id == race_id:
                return rp
        return None

    def set_name(self, name: str):
        self.name = name

    def set_trainer(self, trainer: str):
        self.trainer = trainer

    def set_pedigree(self, sire: 'Dog', dam: 'Dog'):
        self.sire = sire
        self.dam = dam

    def __repr__(self):
        return f"Dog({self.id}, {self.name})"
    
    def print_info(self):
        print({
            "id": self.id,
            "name": self.name,
            "birth_date": self.birth_date,
            "trainer": self.trainer,
            "color": self.color,
            "weight": self.weight,
            "sire": self.sire.id if self.sire else None,
            "dam": self.dam.id if self.dam else None,
            "participations_count": len(self.race_participations)
            })
