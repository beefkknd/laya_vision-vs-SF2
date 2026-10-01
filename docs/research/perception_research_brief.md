Research task (use web search; cite sources with URLs; be concrete; plain markdown).

Context: a bot plays Super Street Fighter II (SNES / arcade-era SF2, CPU opponents). Its "System 1" is a small vision
model - the eye and the muscle memory: it sees the screen (two frames a few frames apart) and must capture the moment and
react from experience, the way a trained player does without thinking. Opponent-specific knowledge lives elsewhere (a
language model that studies each opponent); the vision layer must stay GENERAL (no character names, no special-move
names - only things that hold for any opponent).

Question 1: What do experienced Street Fighter II players say they WATCH and REACT TO, moment to moment? How do they
describe their muscle memory, attention and reading of the screen (e.g. spacing/footsies, whiff punishing, anti-airing
jumps, reacting to projectiles, recovery frames, hit-confirming, throw range, corner position, reading startup
animations, health/meter, dizzy, wake-up/okizeme, conditioning)? Prefer what players themselves say (guides,
interviews, forums, coaching material, SF2-era and classic-SF2 community sites such as Shoryuken / SRK wiki /
Sonichurricane / Eventhubs / Reddit r/StreetFighter / supercombo.gg) over generic advice. Separate what is about the
screen moment (perception, reaction) from what is about knowing a specific character (matchup knowledge).

Question 2: Compare with the questions the vision model will be asked at every decision (it answers from the screen
only):
 1. How far away is he? (close / mid / far)
 2. Is he attacking right now? (yes / no)
 3. Is he in the air? (yes / no)
 4. If I do move X now, is it better than walking in? (per move: likely works / may work / likely fails; a soft,
    general answer - opponent-specific choice is left to the language model)
 5. How full is my health bar, and his? (full / high / half / low)
Do these align with what players attend to? What is missing that players consider essential and that is GENERAL
(holds for any opponent) and visible on screen? What is redundant or mis-framed? Propose a short, prioritized list of
improved or added questions (at most 8 in total), each with: why players care (with a source), whether it is visible on
screen in two frames, and how a label could be derived automatically from the game's RAM at training time. Keep it
general (no character or move names).
End with a one-paragraph recommendation.
