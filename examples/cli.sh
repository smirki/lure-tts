#!/usr/bin/env bash
# Command line tour. Set LURE_MODEL to the model folder first.
set -euo pipefail
REF_WAV=${REF_WAV:-conformance/assets/reference_voice.wav}
REF_TEXT=${REF_TEXT:-"Yes, the weekend order can stay exactly as it is. Everything's fine on our end, truly it is, thank you for checking."}

lure backends                                          # what is installed and what auto picks
lure voice add willow "$REF_WAV" "$REF_TEXT"          # encode once, cached by name
lure say "Hello there. (laugh) Okay." --voice willow -o hello.wav --stats
lure say "Grab the wrench, grab it, grab it!" --voice willow --instruct "Frantic panic, breathless." --seed 4 -o panic.wav
echo "A longer script, read from stdin. It is split into sentences and streamed." | lure say - --voice willow -o script.wav
lure say "Turbo off for comparison." --voice willow --opt turbo=0 -o no_turbo.wav
lure prompt "(sigh) Fine." --direction "Tired." --json | head -c 300; echo
