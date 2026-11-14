"""Frontier exploration over a per-agent coverage map, and coverage sharing between teammates.

Nearest-frontier exploration is the standard baseline for robot exploration (Yamauchi 1997;
coordinated multi-robot variant Burgard et al. 2005; neither is in references.bib yet). The floor
plan is known from the map, so the frontier is simply the reachable floor the agent's coverage map
does not count as searched, and `explore` walks to the nearest such cell by static BFS steps,
retargeting as perception marks cells searched.

Each agent keeps two maps: the team map (own sightings plus teammates' coverage messages), which
`explore` uses to find new tasks without re-searching what a teammate searched; and its own map,
which `search` uses to find an object it needs. Coverage messages say where a teammate looked, not
what it saw, so an object the agent never saw can only be found by its own sight.

The coverage map and its sharing are part of the perception layer (multi-robot map merging), the
same for every condition: a coverage message updates the receiver's map and never creates a
decision point (hard rule 2).
"""

from collections import deque

from sim.agent import ActionResult, AgentState
from sim.config import EXPLORING_TOOLS
from sim.pathing import MOVE_OFFSETS
from sim.world import Cell

COVERAGE_KIND = "coverage"
ENTRY_SEPARATOR = ";"
FIELD_SEPARATOR = ","


def is_searched(searched: dict[Cell, int], cell: Cell, tick: int, ttl: int | None) -> bool:
    """Whether a cell counts as searched: seen, and (with a ttl) seen less than `ttl` ticks ago."""
    seen = searched.get(cell)
    return seen is not None and (ttl is None or tick - seen < ttl)


def mark_searched(agent: AgentState, cells: frozenset[Cell], tick: int, ttl: int | None) -> None:
    """Records the cells in view now, in both coverage maps and the next coverage broadcast.

    Without a ttl only first sightings matter, so cells already recorded are left alone.
    """
    for cell in sorted(cells):
        if ttl is None and cell in agent.own_searched:
            continue
        agent.own_searched[cell] = tick
        agent.searched[cell] = tick
        agent.unsent_coverage[cell] = tick


def nearest_frontier(
    start: Cell,
    floor: frozenset[Cell],
    furniture: frozenset[Cell],
    searched: dict[Cell, int],
    tick: int,
    ttl: int | None,
) -> Cell | None:
    """The nearest reachable floor cell (BFS over floor, the static metric) still worth a look.

    That is a floor cell not counted searched, or one next to a furniture cell not yet seen: a
    floor cell glimpsed at the edge of view range says nothing about the shelf just beyond it,
    and objects lie on furniture.
    """
    visited = {start}
    queue: deque[Cell] = deque([start])
    while queue:
        cell = queue.popleft()
        unseen_furniture = any(
            (cell[0] + dx, cell[1] + dy) in furniture
            and not is_searched(searched, (cell[0] + dx, cell[1] + dy), tick, ttl)
            for dx, dy in MOVE_OFFSETS
        )
        if unseen_furniture or not is_searched(searched, cell, tick, ttl):
            return cell
        for dx, dy in MOVE_OFFSETS:
            neighbour = (cell[0] + dx, cell[1] + dy)
            if neighbour in floor and neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return None


def retarget_explorers(
    agents: dict[str, AgentState],
    floor: frozenset[Cell],
    furniture: frozenset[Cell],
    tick: int,
    ttl: int | None,
) -> dict[str, ActionResult]:
    """Points every `explore` and `search` skill at its nearest frontier cell before movement.

    `explore` uses the team coverage map and ends when no frontier is left. `search` uses the
    agent's own map. A search started on a map already seen everywhere begins a new round with a
    cleared map, since objects move (a dish carried off, shards on a floor seen before the drop)
    and the object it looks for may now be somewhere it already looked. A search that has swept
    its round ends, so its agent gets a decision point: an object inside a closed container is
    never seen, and in C2/C3 no message interrupts a skill (v5.1.00).

    Returns:
        Results of the explorers that end this tick because no frontier is left, by agent. Those
        agents sit out the movement phase.
    """
    results: dict[str, ActionResult] = {}
    for agent_id in sorted(agents):
        agent = agents[agent_id]
        exploring = agent.skill is not None and agent.skill.call.name in EXPLORING_TOOLS
        if not agent.online or tick < agent.thinking_until or not exploring:
            continue
        searching = agent.skill.call.name == "search"
        coverage = agent.own_searched if searching else agent.searched
        frontier = nearest_frontier(agent.cell, floor, furniture, coverage, tick, ttl)
        if frontier is None and searching and not agent.skill.targets:
            agent.own_searched.clear()
            frontier = nearest_frontier(agent.cell, floor, furniture, agent.own_searched, tick, ttl)
        if frontier is None:
            text = "searched everywhere you can reach" if searching else "nothing left to explore"
            results[agent_id] = ActionResult(agent.skill.call, "ok", text)
        elif agent.skill.targets != (frontier,):
            agent.skill.targets = (frontier,)
            agent.skill.best_steps = None
            agent.skill.stuck_ticks = 0
    return results


def coverage_payload(cells: dict[Cell, int]) -> dict[str, str]:
    """A coverage message body: "x,y,tick" entries joined by ";"."""
    entries = [
        FIELD_SEPARATOR.join((str(x), str(y), str(seen))) for (x, y), seen in sorted(cells.items())
    ]
    return {"cells": ENTRY_SEPARATOR.join(entries)}


def merge_coverage(searched: dict[Cell, int], payload: dict[str, str]) -> None:
    """Merges a teammate's coverage entries into a coverage map; the later sighting wins."""
    for entry in payload["cells"].split(ENTRY_SEPARATOR):
        x, y, seen = entry.split(FIELD_SEPARATOR)
        cell = (int(x), int(y))
        if int(seen) > searched.get(cell, -1):
            searched[cell] = int(seen)


def team_coverage(
    agents: dict[str, AgentState], floor: frozenset[Cell], tick: int, ttl: int | None
) -> float:
    """Share of floor cells that at least one agent counts as searched."""
    covered = {
        cell
        for agent in agents.values()
        for cell in agent.searched
        if cell in floor and is_searched(agent.searched, cell, tick, ttl)
    }
    return len(covered) / len(floor)
