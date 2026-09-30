"""Facts about the BIDSleep dataset (PhysioNet, v1.0.1) shared across the package."""

# Files
CHECKSUM_FILE = "SHA256SUMS.txt"
HR_FILE = "hr.csv"  # no header: unix_time_s, bpm
MOTION_FILE = "motion.csv"  # header: Timestamp,x,y,z (~50 Hz, units of g)
LABELS_FILE = "labels.mat"  # recStart, dreem_label, expert_label
NIGHT_FILES = (HR_FILE, MOTION_FILE, LABELS_FILE)

# Time
EPOCH_S = 30  # one sleep-stage label per 30 s epoch, starting at recStart
RECSTART_TZ = "America/New_York"  # recStart is stored as US Eastern local time

# Sleep stages (5-class + Unknown), as encoded in labels.mat
WAKE, N1, N2, N3, REM, UNKNOWN = range(6)
STAGE_NAMES = {WAKE: "Wake", N1: "N1", N2: "N2", N3: "N3", REM: "REM", UNKNOWN: "Unknown"}

# 4-class scheme: Wake / Light (N1+N2) / Deep (N3) / REM. Unknown keeps its code and is masked.
TO_4CLASS = {WAKE: 0, N1: 1, N2: 1, N3: 2, REM: 3, UNKNOWN: UNKNOWN}
STAGE4_NAMES = {0: "Wake", 1: "Light", 2: "Deep", 3: "REM", UNKNOWN: "Unknown"}
