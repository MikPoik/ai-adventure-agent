from typing import List, Optional

from pydantic import BaseModel, Field

from schema.objects import Item


class Character(BaseModel):
    name: Optional[str] = Field("Ted", description="The name of the character.")

    description: Optional[str] = Field(
        "DC", description="The description of the character."
    )
    background: Optional[str] = Field(
        "From DC", description="The background of the character."
    )
    inventory: Optional[List[Item]] = Field(
        [], description="The inventory of the character."
    )
    motivation: Optional[str] = Field(
        "Go to DC", description="The motivation of the character."
    )


class NpcCharacter(Character):
    category: Optional[str] = Field(
        "conversational",
        description="The kind of NPC. Can be 'conversational' or 'merchant'",
    )
    disposition_toward_player: Optional[int] = Field(
        1,
        description="The disposition of the Npc toward the player. 1=Doesn't know you. 5=Knows you very well.",
    )


class Merchant(NpcCharacter):
    """Intent:
    - The Merchant appears in your camp after your second quest.
    - The Merchant updates items after each quest, the cost of the items scale with the player RANK. Higher rank is
      more expensive items.
    - If you buy, disposition goes up.
    - If you only sell, dispotition goes down. Or something?
    """

    pass


class TravelingMerchant(NpcCharacter):
    """Intent:
    - The TravelingMerchant only shows up if your RANK % 5 = 0.
    - They always have REALLY good stuff.
    """

    pass


class HumanCharacter(Character):
    rank: Optional[int] = Field(
        1, description="The rank of a player. Higher rank equals more power."
    )
    gold: Optional[int] = Field(
        0,
        description="The gold the player has. Gold can be used buy items from the Merchant. Gold is acquired by selling items to the Merchant and after every quest.",
    )
    energy: Optional[int] = Field(
        100,
        description="The energy the player has. Going on a quest requires and expends energy. This is the unit of monetization for the game.",
    )