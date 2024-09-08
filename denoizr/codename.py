# fmt: off
import random

# 50 adjectives
adjectives = [
    "brave", "swift", "quick", "happy", "eager",
    "fuzzy", "great", "lucky", "proud", "quiet",
    "sharp", "solid", "sweet", "tough", "vivid",
    "witty", "zesty", "agile", "basic", "breezy",
    "chilly", "crazy", "dizzy", "early", "fancy",
    "fiery", "flash", "flaky", "funny", "giant",
    "happy", "hasty", "honey", "icy", "jazzy",
    "juicy", "lucid", "merry", "noisy", "pale",
    "quick", "rusty", "salty", "sandy", "shady",
    "silly", "sunny", "tight", "unfit", "vital"
]

# 50 nouns
nouns = [
    "eagle", "tiger", "horse", "snake", "shark",
    "whale", "zebra", "panda", "koala", "sheep",
    "beach", "apple", "heart", "ocean", "river",
    "house", "chair", "clock", "table", "cloud",
    "stone", "grass", "light", "music", "plant",
    "crown", "dream", "fairy", "ghost", "tulip",
    "piano", "phone", "queen", "space", "sword",
    "earth", "angel", "flood", "couch", "power",
    "sugar", "bunny", "frost", "puppy", "dream",
    "storm", "fruit", "globe", "flame", "heart"
]

# 50 verbs
verbs = [
    "drive", "write", "climb", "dance", "laugh",
    "serve", "paint", "think", "teach", "carry",
    "dream", "smile", "catch", "sleep", "fight",
    "learn", "sweep", "break", "plant", "throw",
    "watch", "clash", "clean", "float", "glide",
    "guide", "jog", "leap", "reach", "shake",
    "sneak", "speak", "swim", "swing", "twist",
    "shine", "stand", "steal", "swear", "trade",
    "train", "trust", "waste", "whine", "write",
    "blast", "blink", "boost", "burst", "charm"
]

def codename():
    return f"{random.choice(adjectives)}-{random.choice(nouns)}-{random.choice(verbs)}"
