from typing import List, Optional

from pydantic import BaseModel, Field

from schema.objects import Item


class Character(BaseModel):
    name: Optional[str] = Field(None, description="The name of the character.")

    tagline: Optional[str] = Field(
        None, description="A short tagline for your character."
    )
    description: Optional[str] = Field(
        None, description="The description of the character."
    )
    background: Optional[str] = Field(
        None, description="The background of the character."
    )
    inventory: Optional[List[Item]] = Field(
        [], description="The inventory of the character."
    )
    inventory_last_updated: Optional[str] = Field(
        None, description="The timestamp of the last update of the inventory"
    )
    image: Optional[str] = Field(
        default=None, description="The URL for the character image"
    )
    personality: Optional[str] = Field(
        None, description="The personality of the character."
    )
    appearance: Optional[str] = Field(
        None, description="The appearance of the character."
    )
    seed_message: Optional[str] = Field(
        None, description="The seed message for the character."
    )

    def fetch_inventory(self, references: List[str]) -> List[Item]:
        """Fetches inventory that matches the provided names."""
        return [i for i in self.inventory or [] if i.name in references]

    def update_from_web(self, other: "Character"):
        """Performs a gentle update so that the website doesn't accidentally blast over this if it diverges in structure."""
        if other.name:
            self.name = other.name
        if other.description:
            self.description = other.description
        if other.background:
            self.background = other.background
        if other.inventory:
            self.inventory = other.inventory
        if other.personality:
            self.personality = other.personality

    def inventory_description(self) -> str:
        result = f"{self.name} has the following items in their inventory:"
        for item in self.inventory:
            result += f"\n\nname: {item.name}"
            result += f"\ndescription: {item.description}"
        return result

    def is_onboarding_complete(self) -> bool:
        """Return True if the player onboarding has been completed.

        This is used by api.py to decide whether to route to the ONBOARDING AGENT.
        """
        return (
            self.name is not None
            and self.description is not None
            and self.background is not None
            and self.inventory is not None
            # and len(self.inventory) > 0
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
    motivation: Optional[str] = Field(
        "",
        description="The motivation of the Npc. Can be 'business', 'social', or 'other'."
    )


class Merchant(NpcCharacter):
    """Intent:
    - The Merchant appears in your camp after your second quest.
    - The Merchant updates items after each quest, the cost of the items scale with the player RANK. Higher rank is more expensive items.
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
    max_energy: Optional[int] = Field(
        100,
        description="DO NOT USE. DEPRECATED. The maximum energy the player can ever have.",
    )
    """The `max_energy` field is still in the data model so Pydantic doesn't trip on its presence, but it is not used."""

    def update_from_web(self, other: "HumanCharacter"):
        """Performs a gentle update so that the website doesn't accidentally blast over this if it diverges in structure."""
        super().update_from_web(other)
        if other.rank:
            self.rank = other.rank
        if other.gold:
            self.gold = other.gold
        if other.max_energy:
            self.max_energy = other.max_energy
        if other.energy:
            self.energy = other.energy
