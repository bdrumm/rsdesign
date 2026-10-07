"""Shared vocabulary for the self-test corpora (module E).

Everything here is *data*: the Material 3 baseline light scheme (values copied from
``@material/web/tokens/versions/v0_192``), the typescale, shape scale, elevation shadows,
Google-app-like word lists, and the CSS used by the hand-styled M3 surfaces that Material Web
does not ship (top app bar, navigation bar/rail, cards, snackbar, badge).

Both ``mwc_corpus`` (HTML pages) and ``synth`` (pure IR) draw from these tables so that the
two corpora share one colour/typography vocabulary.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from dt.ir import Color, Shadow, TextStyle

# --------------------------------------------------------------------------- md.sys.color (light)
#: Material 3 baseline light scheme (md-ref-palette tones per ``_md-sys-color.scss`` values-light).
M3_COLORS: dict[str, str] = {
    "primary": "#6750a4",
    "on-primary": "#ffffff",
    "primary-container": "#eaddff",
    "on-primary-container": "#21005d",
    "secondary": "#625b71",
    "on-secondary": "#ffffff",
    "secondary-container": "#e8def8",
    "on-secondary-container": "#1d192b",
    "tertiary": "#7d5260",
    "on-tertiary": "#ffffff",
    "tertiary-container": "#ffd8e4",
    "on-tertiary-container": "#31111d",
    "error": "#b3261e",
    "on-error": "#ffffff",
    "error-container": "#f9dedc",
    "on-error-container": "#410e0b",
    "background": "#fef7ff",
    "on-background": "#1d1b20",
    "surface": "#fef7ff",
    "on-surface": "#1d1b20",
    "surface-variant": "#e7e0ec",
    "on-surface-variant": "#49454f",
    "surface-dim": "#ded8e1",
    "surface-bright": "#fef7ff",
    "surface-container-lowest": "#ffffff",
    "surface-container-low": "#f7f2fa",
    "surface-container": "#f3edf7",
    "surface-container-high": "#ece6f0",
    "surface-container-highest": "#e6e0e9",
    "surface-tint": "#6750a4",
    "inverse-surface": "#322f35",
    "inverse-on-surface": "#f5eff7",
    "inverse-primary": "#d0bcff",
    "outline": "#79747e",
    "outline-variant": "#cac4d0",
    "shadow": "#000000",
    "scrim": "#000000",
}


def color(role: str) -> Color:
    """Return the baseline light-scheme colour for an ``md.sys.color`` role (e.g. ``"primary"``)."""
    return Color.from_hex(M3_COLORS[role])


def token_name(role: str) -> str:
    """Design-token name for a colour role, e.g. ``"md.sys.color.primary"``."""
    return f"md.sys.color.{role}"


# --------------------------------------------------------------------------- md.sys.typescale
@dataclass(frozen=True)
class TypeRole:
    """One typescale role: size / line-height / weight / tracking, all in px."""
    name: str
    size: float
    line_height: float
    weight: int
    tracking: float

    def style(self, role_color: str = "on-surface", align: str = "left") -> TextStyle:
        return TextStyle(
            family="Roboto", size=self.size, weight=self.weight, line_height=self.line_height,
            letter_spacing=self.tracking, color=color(role_color), align=align,  # type: ignore[arg-type]
        )


#: The 15 M3 typescale roles (rem values from the token files multiplied by 16).
M3_TYPESCALE: dict[str, TypeRole] = {
    r.name: r
    for r in (
        TypeRole("display-large", 57, 64, 400, -0.25),
        TypeRole("display-medium", 45, 52, 400, 0),
        TypeRole("display-small", 36, 44, 400, 0),
        TypeRole("headline-large", 32, 40, 400, 0),
        TypeRole("headline-medium", 28, 36, 400, 0),
        TypeRole("headline-small", 24, 32, 400, 0),
        TypeRole("title-large", 22, 28, 400, 0),
        TypeRole("title-medium", 16, 24, 500, 0.15),
        TypeRole("title-small", 14, 20, 500, 0.1),
        TypeRole("body-large", 16, 24, 400, 0.5),
        TypeRole("body-medium", 14, 20, 400, 0.25),
        TypeRole("body-small", 12, 16, 400, 0.4),
        TypeRole("label-large", 14, 20, 500, 0.1),
        TypeRole("label-medium", 12, 16, 500, 0.5),
        TypeRole("label-small", 11, 16, 500, 0.5),
    )
}

# --------------------------------------------------------------------------- md.sys.shape
M3_SHAPE: dict[str, float] = {
    "none": 0, "extra-small": 4, "small": 8, "medium": 12, "large": 16, "extra-large": 28, "full": 9999,
}

# --------------------------------------------------------------------------- md.sys.elevation
#: Box shadows per elevation level (the two-layer shadows Material Web's md-elevation paints).
M3_ELEVATION: dict[int, list[Shadow]] = {
    0: [],
    1: [Shadow(Color(0, 0, 0, 0.3), 0, 1, 2, 0), Shadow(Color(0, 0, 0, 0.15), 0, 1, 3, 1)],
    2: [Shadow(Color(0, 0, 0, 0.3), 0, 1, 2, 0), Shadow(Color(0, 0, 0, 0.15), 0, 2, 6, 2)],
    3: [Shadow(Color(0, 0, 0, 0.3), 0, 1, 3, 0), Shadow(Color(0, 0, 0, 0.15), 0, 4, 8, 3)],
    4: [Shadow(Color(0, 0, 0, 0.3), 0, 2, 3, 0), Shadow(Color(0, 0, 0, 0.15), 0, 6, 10, 4)],
    5: [Shadow(Color(0, 0, 0, 0.3), 0, 4, 4, 0), Shadow(Color(0, 0, 0, 0.15), 0, 8, 12, 6)],
}

# --------------------------------------------------------------------------- vocabulary
APP_TITLES = ["Gmail", "Calendar", "Drive", "Photos", "Keep", "Tasks", "Meet", "Chat", "Docs", "Maps"]
SECTION_TITLES = [
    "Inbox", "Starred", "Snoozed", "Sent", "Drafts", "Important", "Shared with me", "Recent",
    "Trash", "Today", "This week", "Upcoming", "Labels", "Settings", "Notifications", "Storage",
]
PEOPLE = [
    "Alex Rivera", "Priya Natarajan", "Jordan Lee", "Sam Okafor", "Taylor Kim", "Morgan Blake",
    "Casey Nguyen", "Riley Santos", "Dana Fischer", "Avery Patel", "Jamie Cole", "Noor Haddad",
]
SUBJECTS = [
    "Q3 planning notes", "Lunch on Thursday?", "Design review follow-up", "Trip itinerary",
    "Invoice #4821", "Photos from the weekend", "Your order has shipped", "Weekly digest",
    "Team offsite agenda", "Re: Budget approval", "Flight confirmation", "Shared: Roadmap 2025",
]
SNIPPETS = [
    "Hi team, here are the notes from today's meeting.",
    "Let me know if that time still works for you.",
    "Attached is the latest draft for your review.",
    "Reminder: the deadline is end of day Friday.",
    "Thanks for sending this over, looks great!",
    "Can we move this to next week instead?",
    "The files are in the shared folder now.",
    "Please confirm your availability by Monday.",
]
FILES = [
    "Roadmap 2025", "Budget.xlsx", "Onboarding guide", "Brand assets", "Meeting notes",
    "Pitch deck v3", "Expense report", "Screenshots", "Travel plan", "Reading list",
]
EVENTS = [
    "Standup", "1:1 with Priya", "Design sync", "Dentist", "Lunch with Jordan", "Sprint review",
    "Yoga class", "Team offsite", "Flight to SFO", "Book club",
]
TIMES = ["9:00 AM", "9:30 AM", "10:15 AM", "11:00 AM", "12:30 PM", "2:00 PM", "3:45 PM", "5:30 PM", "Yesterday", "Mon", "Tue", "Oct 3"]
ACTIONS = ["Save", "Cancel", "Send", "Share", "Done", "Reply", "Archive", "Delete", "Create", "Add", "Join", "Skip", "Next", "Undo", "Dismiss", "Retry", "OK", "Got it"]
CHIP_LABELS = ["Unread", "Starred", "Attachments", "From me", "Calendar", "Docs", "Sheets", "Images", "Last week", "Work", "Personal", "Travel", "Finance"]
FIELD_LABELS = ["Email", "Name", "Subject", "Location", "Search", "Password", "Phone", "Title", "Notes", "Website"]
FIELD_VALUES = ["alex@example.com", "Priya Natarajan", "Weekly sync", "Room 4B", "", "", "+1 415 555 0134", "Roadmap", "", "example.com"]
SETTINGS = [
    "Dark theme", "Notifications", "Sync contacts", "Smart compose", "Offline mode", "Auto-advance",
    "Conversation view", "Show snippets", "Vacation responder", "Desktop alerts", "Location history",
]
MENU_ITEMS = ["Open", "Rename", "Share", "Move to", "Add to starred", "Make a copy", "Download", "Remove"]
DIALOG_TITLES = ["Discard draft?", "Delete 3 items?", "Allow location access?", "Leave meeting?", "Save changes?"]
DIALOG_BODY = [
    "This action cannot be undone.",
    "Items in the trash are deleted after 30 days.",
    "Maps needs your location to show nearby places.",
    "You can rejoin from the calendar invitation.",
]
NAV_DESTS = [
    ("mail", "Mail"), ("chat", "Chat"), ("videocam", "Meet"), ("calendar_month", "Calendar"),
    ("home", "Home"), ("search", "Search"), ("folder_shared", "Shared"), ("star", "Starred"),
    ("person", "People"), ("photo_library", "Photos"), ("checklist", "Tasks"), ("notifications", "Alerts"),
]
LEADING_ICONS = ["mail", "folder", "description", "image", "event", "person", "drafts", "label", "attach_file", "videocam", "place", "note"]
ACTION_ICONS = ["search", "more_vert", "settings", "share", "edit", "delete", "add", "filter_list", "refresh", "close", "arrow_back", "menu", "star", "archive", "send", "check"]
FAB_ICONS = ["add", "edit", "mail", "navigation", "videocam", "mic"]
SNACKBAR_TEXT = ["Message archived", "Draft saved", "3 items moved to Trash", "Event added", "Copied link"]
TABS = [["Primary", "Social", "Promotions"], ["Upcoming", "Past"], ["Mine", "Shared", "Starred", "Recent"], ["Day", "Week", "Month"]]
SELECT_LABELS = [("Time zone", ["Pacific Time", "Eastern Time", "Central European Time"]), ("Repeat", ["Does not repeat", "Daily", "Weekly on Monday"]), ("Visibility", ["Default", "Public", "Private"])]
HEADLINES = ["Welcome back", "Your storage", "Recent activity", "Suggested for you", "Quick access", "Upcoming events", "Shared with you"]


def pick(rng: random.Random, xs: list, k: int = 1) -> list:
    """Deterministic sample of ``k`` distinct items (fewer if ``xs`` is short)."""
    k = min(k, len(xs))
    return rng.sample(xs, k)


# --------------------------------------------------------------------------- CSS for hand-styled surfaces
def root_css_vars() -> str:
    """``--md-sys-color-*`` custom properties for the baseline light scheme, set on ``:root``."""
    lines = [f"  --md-sys-color-{k}: {v};" for k, v in M3_COLORS.items()]
    return ":root {\n" + "\n".join(lines) + "\n}"


#: CSS for the surfaces Material Web lacks. Metrics follow the M3 specs (top app bar 64px,
#: navigation bar 80px with 64x32 pill indicators, rail 80px wide with 56x32 indicators, cards
#: 12px radius, snackbar 48px min height 4px radius, badge 16px/6px).
HAND_STYLED_CSS = """
.app-bar { display:flex; align-items:center; height:64px; padding:0 4px; background:var(--md-sys-color-surface); color:var(--md-sys-color-on-surface); gap:4px; box-sizing:border-box; }
.app-bar.center { justify-content:space-between; }
.app-bar.center .title { flex:1; text-align:center; }
.app-bar .title { font-size:22px; line-height:28px; font-weight:400; letter-spacing:0; padding:0 12px; white-space:nowrap; overflow:hidden; }
.app-bar.small .title { flex:1; }
.app-bar .spacer { flex:1; }
.app-bar.container { background:var(--md-sys-color-surface-container); }
.nav-bar { display:flex; height:80px; background:var(--md-sys-color-surface-container); box-sizing:border-box; padding:12px 8px 16px; gap:8px; }
.nav-bar .dest { flex:1; display:flex; flex-direction:column; align-items:center; gap:4px; color:var(--md-sys-color-on-surface-variant); }
.nav-bar .dest .pill { width:64px; height:32px; border-radius:16px; display:flex; align-items:center; justify-content:center; }
.nav-bar .dest.active .pill { background:var(--md-sys-color-secondary-container); }
.nav-bar .dest.active { color:var(--md-sys-color-on-surface); }
.nav-bar .dest .lbl { font-size:12px; line-height:16px; font-weight:500; letter-spacing:0.5px; }
.nav-bar .dest.active .lbl { font-weight:700; }
.nav-rail { width:80px; display:flex; flex-direction:column; align-items:center; background:var(--md-sys-color-surface); padding:12px 0; gap:12px; box-sizing:border-box; flex:none; }
.nav-rail .dest { display:flex; flex-direction:column; align-items:center; gap:4px; color:var(--md-sys-color-on-surface-variant); }
.nav-rail .dest .pill { width:56px; height:32px; border-radius:16px; display:flex; align-items:center; justify-content:center; }
.nav-rail .dest.active .pill { background:var(--md-sys-color-secondary-container); }
.nav-rail .dest .lbl { font-size:12px; line-height:16px; font-weight:500; letter-spacing:0.5px; }
.nav-rail .dest.active { color:var(--md-sys-color-on-surface); }
.card { border-radius:12px; padding:16px; box-sizing:border-box; display:flex; flex-direction:column; gap:8px; color:var(--md-sys-color-on-surface); }
.card.elevated { background:var(--md-sys-color-surface-container-low); box-shadow:0px 1px 2px 0px rgba(0,0,0,0.3), 0px 1px 3px 1px rgba(0,0,0,0.15); }
.card.filled { background:var(--md-sys-color-surface-container-highest); }
.card.outlined { background:var(--md-sys-color-surface); border:1px solid var(--md-sys-color-outline-variant); }
.card .headline { font-size:16px; line-height:24px; font-weight:500; letter-spacing:0.15px; }
.card .supporting { font-size:14px; line-height:20px; letter-spacing:0.25px; color:var(--md-sys-color-on-surface-variant); }
.card .actions { display:flex; gap:8px; justify-content:flex-end; margin-top:8px; }
.card .media { height:120px; border-radius:8px; background:var(--md-sys-color-primary-container); }
.snackbar { position:absolute; left:16px; right:16px; min-height:48px; border-radius:4px; background:var(--md-sys-color-inverse-surface); color:var(--md-sys-color-inverse-on-surface); display:flex; align-items:center; padding:0 8px 0 16px; box-sizing:border-box; gap:8px; box-shadow:0px 1px 3px 0px rgba(0,0,0,0.3), 0px 4px 8px 3px rgba(0,0,0,0.15); }
.snackbar .msg { flex:1; font-size:14px; line-height:20px; letter-spacing:0.25px; }
.snackbar .act { color:var(--md-sys-color-inverse-primary); font-size:14px; line-height:20px; font-weight:500; letter-spacing:0.1px; padding:10px 12px; }
.badge-wrap { position:relative; display:inline-flex; }
.badge { position:absolute; top:-2px; right:-4px; min-width:16px; height:16px; padding:0 4px; box-sizing:border-box; border-radius:8px; background:var(--md-sys-color-error); color:var(--md-sys-color-on-error); font-size:11px; line-height:16px; font-weight:500; letter-spacing:0.5px; text-align:center; }
.badge.small { width:6px; min-width:6px; height:6px; padding:0; border-radius:3px; top:0; right:0; }
.section-title { font-size:14px; line-height:20px; font-weight:500; letter-spacing:0.1px; color:var(--md-sys-color-on-surface-variant); padding:16px 16px 8px; }
.headline-text { font-size:24px; line-height:32px; font-weight:400; color:var(--md-sys-color-on-surface); padding:16px 16px 4px; }
.body-text { font-size:14px; line-height:20px; letter-spacing:0.25px; color:var(--md-sys-color-on-surface-variant); padding:0 16px 8px; }
.row { display:flex; align-items:center; gap:8px; padding:8px 16px; flex-wrap:wrap; }
.row.end { justify-content:flex-end; }
.grid { display:grid; gap:12px; padding:8px 16px; }
.form { display:flex; flex-direction:column; gap:16px; padding:12px 16px; }
.progress-row { display:flex; align-items:center; gap:16px; padding:8px 16px; }
.fab-holder { position:absolute; right:16px; }
.menu-anchor { position:relative; display:inline-block; }
.avatar { width:40px; height:40px; border-radius:20px; background:var(--md-sys-color-primary-container); color:var(--md-sys-color-on-primary-container); display:flex; align-items:center; justify-content:center; font-size:16px; font-weight:500; }
"""


def elevation_css(level: int) -> str:
    """CSS ``box-shadow`` value for an M3 elevation level."""
    return ", ".join(f"{int(s.dx)}px {int(s.dy)}px {int(s.blur)}px {int(s.spread)}px {s.color.css()}" for s in M3_ELEVATION[level]) or "none"
