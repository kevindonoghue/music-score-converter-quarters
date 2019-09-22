import numpy as np

treble_notes = ['G3', 'A3', 'B3'] + [f'{ch}4' for ch in list('CDEFGAB')] + [f'{ch}5' for ch in list('CDEFGAB')] + ['C6', 'D6', 'E6', 'F6', 'G6', 'A6', 'B6', 'C7']
bass_notes = ['F1', 'G1', 'A1', 'B1'] + [f'{ch}2' for ch in list('CDEFGAB')] + [f'{ch}3' for ch in list('CDEFGAB')] + [f'{ch}4' for ch in list('CDEF')]

start_probs = dict()
start_probs['treble'] = dict()
start_probs['bass'] = dict()

for i, x in enumerate(bass_notes):
    if i < bass_notes.index('E2'):
        start_probs['bass'][x] = 1
    elif i > bass_notes.index('C4'):
        start_probs['bass'][x] = 1
    else:
        start_probs['bass'][x] = 10

bass_total = np.sum([x for _, x in start_probs['bass'].items()])
for note in start_probs['bass']:
    start_probs['bass'][note] /= bass_total

for i, x in enumerate(treble_notes):
    if i < treble_notes.index('C4'):
        start_probs['treble'][x] = 1
    elif i > treble_notes.index('A5'):
        start_probs['treble'][x] = 1
    else:
        start_probs['treble'][x] = 10

treble_total = np.sum([x for _, x in start_probs['treble'].items()])
for note in start_probs['treble']:
    start_probs['treble'][note] /= treble_total

# ptp stands for pitch transition probabilities:
ptp = dict()
for i in range(-5, 6):
    ptp[i] = 60
for i in list(range(-8, -5)) + list(range(6, 9)):
    ptp[i] = 5
for i in list(range(-10, -8)) + list(range(9, 11)):
    ptp[i] = 1

ptp_total = np.sum([x for _, x in ptp.items()])
for interval in ptp:
    ptp[interval] /= ptp_total

start_probs_items = dict()
start_probs_items['treble'] = start_probs['treble'].items()
start_probs_items['bass'] = start_probs['bass'].items()

ptp_items = ptp.items()



def generate_chords(num_chords, clef, chord_probs):
    # clef is either 'treble' or 'bass'
    # chord_probs is an array of length 15 that gives the probabilities of adding notes to make chords
    # the entries of chord_probs correspond to adding an octave below to an octave above
    pitches = [x for x, _ in start_probs_items[clef]]
    start_probs = [y for _, y in start_probs_items[clef]]
    start_pitch = np.random.choice(pitches, p=start_probs)
    intervals = [x for x, _ in ptp_items]
    interval_probs = [y for _, y in ptp_items]

    generated_pitches = [[start_pitch]]
    while len(generated_pitches) < num_chords:
        reference_pitch = generated_pitches[-1][-1]

        interval = np.random.choice(intervals, p=interval_probs)
        last_pitch_ix = pitches.index(reference_pitch)
        if last_pitch_ix + interval in range(len(pitches)):
            generated_pitches.append([pitches[last_pitch_ix + interval]])

    for i, x in enumerate(generated_pitches):
        ix = pitches.index(x[0])
        for j, p in enumerate(chord_probs):
            if np.random.rand() < p and j != 7:
                interval = -7 + j
                if ix + interval in range(len(pitches)):
                    generated_pitches[i].append(pitches[ix + interval])

    return generated_pitches
