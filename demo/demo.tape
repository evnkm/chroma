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
Sleep 5s

# Query 1: realistic predicate, dramatic.
Type "how does photosynthesis work"
Enter
Sleep 500ms
# Pick preset 1: source_domain = wikihow.com.
Type "1"
Enter
Sleep 4s

# Query 2: synthetic 0.1% killer case.
Type "what causes ocean tides"
Enter
Sleep 500ms
# Pick preset 4: synthetic topic_bucket = 0.
Type "4"
Enter
Sleep 4s

# Quit.
Type "q"
Enter
Sleep 1s
