"""The built-in skirmish AI's rules, as a teacher over MacroEnv's actions.

Expert.recommend() returns the actions the game's own AI would take in the agent's place,
following HouseClass::AI_Building, AI_Unit, AI_Infantry and Expert_AI / AI_Attack in
tiberiandawn/house.cpp with the default rules.ini values:

- Buildings: every candidate gets an urgency from the base composition (power surplus,
  refineries up to 18% of the buildings, barracks, a war factory once there are a few
  refineries, guard towers up to half the buildings, ...) and the most urgent ones are
  recommended, as the AI picks at random among them.
- Vehicles: a harvester while there are more refineries than harvesters, otherwise any
  armed vehicle.
- Infantry: riflemen, grenadiers (or flamethrowers) and rocket soldiers while there are
  fewer infantry than buildings or more than 3,000 credits to spend.
- Attack: from the first minute, every 1.5-6 minutes, send the army to hunt (MacroEnv's
  hunt, the same search-and-destroy mission the AI uses).

It reads only what the agent observes, and only recommends actions the mask allows. The
AI's hidden scenario state (teams, IQ, the house it has picked as its enemy) is left out.

variant="improved" adds rules the game's AI code plans but never finished, and avoids the
money lock-up that the faithful rules run into when replayed through MacroEnv:

- Power first (Check_Build_Power, active in Red Alert, a stub in TD): high urgency whenever
  the surplus is 50 or less, critical when short, low only before the first refinery.
- Income (Check_Build_Income, only a comment in the game): refineries are high urgency
  while there are fewer than two, medium while money is under 1,000, low otherwise; and
  two are wanted whatever the base size, since with one harvester the base never reaches
  the six buildings the 18% ratio needs for a second.
- Barracks medium rather than high when there are none, as in Red Alert.
- Only start what is affordable now. When the most urgent building isn't, save for it:
  recommend no building, and only units that leave its cost in the bank (harvesters
  excepted). The faithful rules start it on credit whenever there is any income, and TD
  then drains money into it as it builds, starving everything else.
"""

import math

import numpy as np

from ._native import AIRCRAFT, BUILDING, ENEMY, INFANTRY, SELF

TICKS_PER_MINUTE = 15 * 60

# Rules defaults (tiberiandawn/rules.cpp).
POWER_SURPLUS = 50
REFINERY_RATIO, REFINERY_LIMIT = 0.18, 7
BARRACKS_RATIO, BARRACKS_LIMIT = 0.16, 2
WAR_RATIO, WAR_LIMIT = 0.1, 3
DEFENSE_RATIO, DEFENSE_LIMIT = 0.5, 25
TESLA_RATIO, TESLA_LIMIT = 0.8, 5
HELIPAD_RATIO, HELIPAD_LIMIT = 0.12, 6
AA_RATIO, AA_LIMIT = 0.14, 10
INFANTRY_RESERVE = 3000
ATTACK_INTERVAL = 3  # Times half a minute to two minutes between attacks.

LOW, MEDIUM, HIGH, CRITICAL = 1, 2, 3, 4
VARIANTS = ("ai", "improved")

ARMED_VEHICLES = ("light_vehicle", "bike", "apc", "tank", "heavy_tank", "artillery", "stealth_tank",
                  "rocket_launcher")
INFANTRY_ROLES = ("rifleman", "grenadier", "rocket_soldier")


def _round_up(ratio, buildings):
    return math.ceil(ratio * buildings - 1e-9)


class Expert:
    """The built-in AI's choices for a MacroEnv; call reset() at each episode start."""

    def __init__(self, macro, variant="ai"):
        if variant not in VARIANTS:
            raise ValueError("expert variant must be one of %s" % (VARIANTS,))
        self.macro = macro
        self.improved = variant == "improved"
        self.reset()

    def reset(self):
        self._next_attack = TICKS_PER_MINUTE
        self._harvesters = {}  # id: last frame seen.
        self._reserve = 0  # Credits being saved for a building, for the improved variant.

    def note_action(self, name, frame):
        """The agent took action 'name' at game frame 'frame': an attack restarts the timer."""
        if name in ("attack", "hunt"):
            rng = self.macro.base.np_random
            self._next_attack = frame + ATTACK_INTERVAL * int(
                rng.integers(TICKS_PER_MINUTE // 2, TICKS_PER_MINUTE * 2 + 1)
            )

    def recommend(self, mask):
        """Groups of recommended action names, most important first: each group is a set the
        AI would pick from at random. Only actions 'mask' allows are included."""
        from .macro import ACTION_INDEX

        allowed = lambda name: bool(mask[ACTION_INDEX[name]])  # noqa: E731
        groups = []
        if allowed("deploy_mcv"):
            groups.append(["deploy_mcv"])
        state = self._state()
        if state["frame"] >= self._next_attack and allowed("hunt"):
            groups.append(["hunt"])
        for choose in (self._building, self._vehicle, self._infantry):
            group = [name for name in choose(state) if allowed(name)]
            if group:
                groups.append(group)
        return groups

    def mask(self, mask):
        """int8 mask over ACTIONS of every recommended action."""
        from .macro import ACTION_INDEX

        out = np.zeros(len(mask), dtype=np.int8)
        for group in self.recommend(mask):
            for name in group:
                out[ACTION_INDEX[name]] = 1
        return out

    def choose(self, mask, rng):
        """One action index the AI would take, or noop."""
        from .macro import ACTION_INDEX

        groups = self.recommend(mask)
        if not groups:
            return ACTION_INDEX["noop"]
        return ACTION_INDEX[groups[0][int(rng.integers(len(groups[0])))]]

    # The state the rules look at.

    def _state(self):
        base = self.macro.base
        objects = base.objects
        own = objects[objects["relation"] == SELF]
        names = [n.decode() for n in own["name"]]
        scalars = base._scalars
        # A harvester unloading inside a refinery isn't on the map, so count the ones seen in
        # the last minute rather than mistake it for lost and build another.
        frame = scalars["frame"]
        for harvester in own[own["name"] == b"HARV"]:
            self._harvesters[int(harvester["id"])] = frame
        self._harvesters = {i: f for i, f in self._harvesters.items() if frame - f <= TICKS_PER_MINUTE}
        names = [n for n in names if n != "HARV"] + ["HARV"] * len(self._harvesters)
        count = lambda *ns: sum(names.count(n) for n in ns)  # noqa: E731
        enemies = objects[objects["relation"] == ENEMY]
        return {
            "frame": scalars["frame"],
            "money": scalars["credits"] + scalars["tiberium"],
            "power": scalars["power_produced"],
            "drain": scalars["power_drained"],
            "buildings": int((own["type"] == BUILDING).sum()),
            "infantry": int((own["type"] == INFANTRY).sum()),
            "count": count,
            "nod": base._episode_side == 1,
            "air_threat": bool(
                (enemies["type"] == AIRCRAFT).any() or np.isin(enemies["name"], (b"HPAD", b"AFLD")).any()
            ),
            "enemy_aircraft": int((enemies["type"] == AIRCRAFT).sum()),
        }

    def _cost(self, role, kind="build_"):
        from .macro import STRUCTURES, UNITS

        names = (STRUCTURES if kind == "build_" else UNITS)[role]
        for name in names:
            b = self.macro.base._buildables.get(name)
            if b is not None:
                return int(b["cost"])
        return None  # Not on the sidebar: can't be built.

    def _building(self, s):
        """AI_Building: the most urgent structures."""
        count, money, b = s["count"], s["money"], s["buildings"]
        refineries = count("PROC")
        income = refineries > 0 and count("HARV") > 0
        improved = self.improved
        choices = []  # (urgency, action, cost)

        def offer(role, urgency, can_afford=None):
            cost = self._cost(role)
            if cost is None:
                return
            if can_afford is None:
                can_afford = cost < money or income
            # The improved rules consider everything, and decide below whether to start it
            # or save for it.
            if improved or can_afford:
                choices.append((urgency, "build_" + role, cost))

        if s["power"] <= s["drain"] + POWER_SURPLUS:
            short = s["power"] < s["drain"]
            urgency = LOW if refineries == 0 else CRITICAL if short else HIGH if improved else MEDIUM
            for role in ("advanced_power", "power"):
                cost = self._cost(role)
                if cost is not None and (cost < money or (improved and role == "power")):
                    offer(role, urgency, cost < money)
                    break
        # One harvester can't grow a base to the 6 buildings the ratio wants for a second
        # refinery, so the improved rules always allow two.
        wanted = max(_round_up(REFINERY_RATIO, b), 2 if improved else 0)
        if refineries < wanted and refineries < REFINERY_LIMIT:
            if improved:
                urgency = HIGH if refineries < 2 else MEDIUM if money < 1000 else LOW
            else:
                urgency = HIGH if refineries < 2 else MEDIUM
            cost = self._cost("refinery")
            offer("refinery", urgency, cost is not None and (money > cost or income))
        current = count("PYLE", "HAND")
        if current < _round_up(BARRACKS_RATIO, b) and current < BARRACKS_LIMIT and (money > 300 or income):
            offer("barracks", LOW if current else MEDIUM if improved else HIGH)
        current = count("WEAP", "AFLD")
        if current < _round_up(WAR_RATIO, b) and current < WAR_LIMIT and (money > 2000 or income):
            if (refineries <= 3 and current < 1) or (refineries >= 5 and current < 2):
                urgency = HIGH
            else:
                urgency = MEDIUM if current == 0 else LOW
            offer("factory", urgency)
        current = count("GTWR", "GUN")
        if current < _round_up(DEFENSE_RATIO, b) and current < DEFENSE_LIMIT:
            offer("defense", MEDIUM)
        powered = s["drain"] == 0 or s["power"] >= s["drain"]
        current = count("ATWR", "OBLI")
        if current < _round_up(TESLA_RATIO, b) and current < TESLA_LIMIT and powered:
            offer("advanced_defense", MEDIUM)
        if not s["nod"] and count("FIX") == 0 and count("WEAP") > 0:
            offer("repair", MEDIUM)
        if s["nod"] and count("HQ") == 0 and count("AFLD") > 0:
            offer("radar", MEDIUM)
        current = count("HPAD")
        if current < _round_up(HELIPAD_RATIO, b) and current < HELIPAD_LIMIT:
            offer("helipad", LOW if current else MEDIUM)
        current = count("SAM", "ATWR")
        if current < _round_up(AA_RATIO, b) and current < AA_LIMIT and s["air_threat"]:
            if not s["nod"] and count("HQ") == 0:
                offer("radar", HIGH)
            urgency = HIGH if current < s["enemy_aircraft"] else MEDIUM
            offer("advanced_defense" if not s["nod"] else "anti_air", urgency)

        self._reserve = 0
        if not choices:
            return []
        best_urgency = max(u for u, _, _ in choices)
        best = [(name, cost) for u, name, cost in choices if u == best_urgency]
        if improved:
            affordable = [(name, cost) for name, cost in best if cost <= money]
            if not affordable:
                self._reserve = min(cost for _, cost in best)
                return []
            best = affordable
        return sorted({name for name, _ in best})

    def _vehicle(self, s):
        """AI_Unit: harvesters to match refineries, then any armed vehicle."""
        count = s["count"]
        if count("PROC") > count("HARV") and self._cost("harvester", "train_") is not None:
            return ["train_harvester"]
        return self._affordable(ARMED_VEHICLES, s)

    def _infantry(self, s):
        """AI_Infantry: basic infantry while short of infantry or with money to spare."""
        if not (s["money"] > INFANTRY_RESERVE or s["infantry"] < s["buildings"]):
            return []
        return self._affordable(INFANTRY_ROLES, s)

    def _affordable(self, roles, s):
        """Units of 'roles' that fit in the money, less any savings for a building."""
        budget = s["money"] - self._reserve
        return ["train_" + role for role in roles if (cost := self._cost(role, "train_")) is not None and cost <= budget]
