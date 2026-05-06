# vhs script to record the SPANN race demo as .gif / .mp4.
# Install: brew install vhs
# Run:     vhs demo/demo.tape
#
# Drives the TUI deterministically: opens it, picks a query + predicate,
# waits for the race to render, then repeats with the synthetic predicate
# for comparison.

Output demo/recordings/spann-race.gif
Output demo/recordings/spann-race.mp4

Set Width 1200
Set Height 720
Set FontSize 14
Set Theme "Catppuccin Mocha"
Set Padding 8

Hide
Type "cd $(git rev-parse --show-toplevel) && source .venv/bin/activate"
Enter
Sleep 500ms
Show

Type "python demo/tui.py"
Enter
Sleep 6s

# Query 1: realistic predicate. "how to" naturally aligns with wikihow.com docs.
Type "how to keep a houseplant alive"
Enter
Sleep 800ms
# Pick preset 1: source_domain = wikihow.com (0.74% selectivity).
Type "1"
Enter
Sleep 5s

# Query 2: synthetic 0.1% killer case (matches paper headline).
Type "what causes ocean tides"
Enter
Sleep 800ms
# Pick preset 4: synthetic topic_bucket = 0 (0.1% selectivity).
Type "4"
Enter
Sleep 5s

# Quit.
Type "q"
Enter
Sleep 1s
