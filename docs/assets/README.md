# Deck assets

The title and header logo is picked up automatically from the first of
these that exists:

    docs/IITD-_logo.png          <- currently in use
    docs/assets/iitd_ee_logo.png
    docs/assets/logo.png

Without any of them the deck falls back to a text lockup, so the build
never breaks.

The IIT Delhi crest is square (220x220 with transparency), and every
placement in `make_deck.js` uses a 1:1 box for that reason. If you swap in
a wide departmental lockup instead, change `d0` in `brand()` and the width
in `head()` to match its aspect ratio - a square box would squash it.
