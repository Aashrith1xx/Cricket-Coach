"""The shot taxonomy, and which axes of it a camera can actually resolve.

Coaching manuals classify batting shots on three independent axes - **footwork**, **bat
plane** and **direction** - plus intent. That structure is what makes classification
tractable: rather than learning twelve appearances, measure three or four properties and read
off which shot they describe.

    footwork    front foot | back foot        which foot the weight goes to
    bat plane   vertical | horizontal         down the line of the ball, or across it
    direction   off | straight | leg          where the ball is sent
    intent      attacking | defensive         committed stroke or a block
    elevation   ground | lofted               hit along the deck or over the top

Crucially, the axes are **not equally measurable**, and the taxonomy is where that gets
recorded. Footwork lives along the pitch, so a side-on camera sees it and a front-on camera
cannot. Direction is the reverse. Bat plane and intent are visible from either.

So the classifier resolves whichever axes the view supports and narrows to the shots
consistent with them. When the surviving set has one member it names the shot; when it has
several it says which family and lists them. That is the honest output of a partial
measurement, and it beats naming one of twelve from evidence that cannot separate them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Tuple

FRONT, BACK = "front", "back"
VERTICAL, HORIZONTAL = "vertical", "horizontal"
OFF, STRAIGHT, LEG = "off", "straight", "leg"
ATTACKING, DEFENSIVE = "attacking", "defensive"
GROUND, LOFTED = "ground", "lofted"


@dataclass(frozen=True)
class Shot:
    key: str
    label: str
    footwork: str
    plane: str
    #: ``None`` where the shot is played to more than one region.
    direction: Optional[str]
    intent: Optional[str]
    elevation: str = GROUND
    #: Extra distinguishing behaviour no axis captures.
    note: str = ""


#: The twelve shots, placed on the axes above.
#:
#: Where two shots share every measurable axis - cover drive and off drive, back-foot drive
#: and punch - they are deliberately kept as one entry rather than split into a distinction
#: no camera here can make.
SHOTS: Tuple[Shot, ...] = (
    Shot("cover_drive", "Cover drive", FRONT, VERTICAL, OFF, ATTACKING),
    Shot("straight_drive", "Straight drive", FRONT, VERTICAL, STRAIGHT, ATTACKING),
    Shot("on_drive_flick", "Flick / on drive", FRONT, VERTICAL, LEG, ATTACKING,
         note="Wristy, worked to the leg side off the front foot."),
    Shot("lofted_drive", "Lofted drive", FRONT, VERTICAL, None, ATTACKING, LOFTED,
         note="Driven with an upward bat path to clear the infield."),
    Shot("defensive_shot", "Front-foot defence", FRONT, VERTICAL, None, DEFENSIVE),
    Shot("advancing", "Advancing down the pitch", FRONT, VERTICAL, None, ATTACKING,
         note="Large forward movement before the stride."),
    Shot("pull_shot", "Pull", BACK, HORIZONTAL, LEG, ATTACKING),
    Shot("cut_shot", "Cut", BACK, HORIZONTAL, OFF, ATTACKING),
    Shot("late_cut", "Late cut", BACK, HORIZONTAL, OFF, ATTACKING,
         note="Played late, close to the body."),
    Shot("back_foot_drive", "Back-foot drive / punch", BACK, VERTICAL, None, ATTACKING),
    Shot("back_foot_defence", "Back-foot defence", BACK, VERTICAL, None, DEFENSIVE),
    Shot("leg_glance", "Leg glance", BACK, VERTICAL, LEG, None,
         note="Deflected fine off the hip."),
)

BY_KEY: Dict[str, Shot] = {s.key: s for s in SHOTS}

#: Which camera view can resolve each axis. The governing rule throughout this project:
#: a quantity is measurable only when it varies in the camera's image plane.
AXIS_VIEWS: Dict[str, FrozenSet[str]] = {
    # Along the pitch - visible from the side, foreshortened from in front.
    "footwork": frozenset({"side_on"}),
    # Across the pitch - the reverse.
    "direction": frozenset({"front_on"}),
    # Both cameras see the bat rise and fall.
    "plane": frozenset({"side_on", "front_on"}),
    "intent": frozenset({"side_on", "front_on"}),
    "elevation": frozenset({"side_on", "front_on"}),
}


def measurable_axes(view: str) -> FrozenSet[str]:
    return frozenset(a for a, views in AXIS_VIEWS.items() if view in views)


def candidates(observed: Dict[str, str]) -> Tuple[Shot, ...]:
    """Shots consistent with the axes that were actually measured.

    An axis a shot leaves open (``None``) matches anything - a lofted drive is lofted
    wherever it is hit - so an unspecified axis never excludes a shot.
    """
    out = []
    for shot in SHOTS:
        for axis, value in observed.items():
            expected = getattr(shot, axis, None)
            if expected is not None and expected != value:
                break
        else:
            out.append(shot)
    return tuple(out)


def family_label(shots: Tuple[Shot, ...]) -> str:
    """A description of what the surviving set has in common."""
    if not shots:
        return "Couldn't tell"
    if len(shots) == 1:
        return shots[0].label

    foot = {s.footwork for s in shots}
    plane = {s.plane for s in shots}
    parts = []
    if len(foot) == 1:
        parts.append("front-foot" if foot == {FRONT} else "back-foot")
    if len(plane) == 1:
        parts.append("vertical-bat" if plane == {VERTICAL} else "horizontal-bat")

    return (" ".join(parts) + " shot").capitalize() if parts else "Batting shot"
