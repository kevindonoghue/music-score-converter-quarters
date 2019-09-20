import numpy as np
from generate_rhythm import produce_subdivision
from generate_chords import generate_chords
from bs4 import BeautifulSoup



def generate_measure(total, key_number, rest_prob, chord_probs, measure_number):
    # total is the total length of the measure in 16th notes
    # clef is either 'treble' or 'bass'
    # key is in range(-7, 8)
    # chord_probs is an array of length 15 that gives the probabilities of adding notes to make chords
    # the entries of chord_probs correspond to adding an octave below to an octave above
    rhythm = dict()
    chords = dict()
    for clef_type in ('treble', 'bass'):
        rhythm[clef_type] = produce_subdivision(total)
        chords[clef_type] = generate_chords(len(rhythm[clef_type]), clef_type, chord_probs)
        for i, r in enumerate(rhythm[clef_type]):
            if r in (1, 2, 4, 8, 16):
                rest_bool = np.random.choice([True, False], p=[rest_prob, 1-rest_prob])
                if rest_bool:
                    chords[clef_type][i] = 'rest'
    
    soup = BeautifulSoup('', 'xml')
    measure = soup.new_tag('measure', number=str(measure_number))

    def note_to_soup(s, dur, rest, staff_number):
        # here s is a string like 'E5'
        # dot and rest are bools
        note = soup.new_tag('note')
        if rest:
            pitch = None
            step = None
            alter = None
            octave = None
            rest = soup.new_tag('rest')
        else:
            pitch = soup.new_tag('pitch')
            step = soup.new_tag('step')
            step.string = s[0]
            alt = str(np.random.choice(['-1', '0', '1'], p=[0.15, 0.7, 0.15]))
            if alt != '0':
                alter = soup.new_tag('alter')
                alter.string = str(alt)
            else:
                alter = None
            octave = soup.new_tag('octave')
            octave.string = s[1]
            rest = None

        if dur == 1:
            type_string = '16th'
            dot = None
        elif dur == 2:
            type_string = 'eighth'
            dot = None
        elif dur == 3:
            type_string = 'eighth'
            dot = soup.new_tag('dot')
        elif dur == 4:
            type_string = 'quarter'
            dot = None
        elif dur == 6:
            type_string = 'quarter'
            dot = soup.new_tag('dot')
        elif dur == 8:
            type_string = 'half'
            dot = None
        elif dur == 12:
            type_string = 'half'
            dot = soup.new_tag('dot')
        elif dur == 16:
            type_string = 'whole'
            dot = None

        duration = soup.new_tag('duration')
        duration.string = str(dur)

        type_ = soup.new_tag('type')
        type_.string = type_string

        if pitch:
            note.append(pitch)
            pitch.append(step)
            if alter:
                pitch.append(alter)
            pitch.append(octave)
            
        if rest:
            note.append(rest)
        note.append(duration)
        note.append(type_)
        if dot: # should always be false for rests
            note.append(dot)
        staff = soup.new_tag('staff')
        note.append(staff)
        staff.string = str(staff_number)
        return note

    def chord_to_soup(x, dur, staff_number):
        # here x is either an array of pitches like ['E5', 'A6'] or the string 'rest'
        # returns a list of note tags
        if x == 'rest':
            return [note_to_soup(None, dur, True, staff_number)]
        else:
            notes = [note_to_soup(s, dur, False, staff_number) for s in x]
            if len(notes) > 1:
                for note in notes[1:]:
                    note.insert(0, soup.new_tag('chord'))
            return notes

    for clef_type in ('treble', 'bass'):
        staff_number = 1 if clef_type == 'treble' else 2
        chords[clef_type] = [chord_to_soup(chord, rhythm[clef_type][i], staff_number) for i, chord in enumerate(chords[clef_type])]
        if np.random.rand() < 0.5 and len(chords[clef_type]) > 1:
            slur_indices = np.random.choice(len(chords[clef_type]), size=2, replace=False)
            initial_index = np.min(slur_indices)
            final_index = np.max(slur_indices)
            initial_note = chords[clef_type][initial_index][0]
            final_note = chords[clef_type][final_index][0]
            if not initial_note.find_all('rest') and not final_note.find_all('rest'): 
                notations = soup.new_tag('notations')
                slur = soup.new_tag('slur')
                slur['type'] = 'start'
                initial_note.append(notations)
                notations.append(slur)
                notations = soup.new_tag('notations')
                slur = soup.new_tag('slur')
                slur['type'] = 'stop'
                final_note.append(notations)
                notations.append(slur)
        for i, chord in enumerate(chords[clef_type]):
            if np.random.rand() < 0.2 and staff_number == 1:
                direction = soup.new_tag('direction')
                direction['placement'] = 'below'
                direction_type = soup.new_tag('direction-type')
                dynamics = soup.new_tag('dynamics')
                dynamic_tag_name = np.random.choice(['ff', 'f', 'mf', 'mp', 'p', 'pp'], p=[0.05, 0.3, 0.15, 0.15, 0.3, 0.05 ])
                dynamic_tag = soup.new_tag(dynamic_tag_name)
                measure.append(direction)
                direction.append(direction_type)
                direction_type.append(dynamics)
                dynamics.append(dynamic_tag)
                staff = soup.new_tag('staff')
                staff.string = '1'
            for note in chord:
                measure.append(note)
        if staff_number == 1:
            backup = soup.new_tag('backup')
            duration = soup.new_tag('duration')
            duration.string = str(total)
            backup.append(duration)
            measure.append(backup)

    return measure


def generate_attributes(measure_length, key_number):
    soup = BeautifulSoup('', 'xml')
    attributes = soup.new_tag('attributes')
    divisions = soup.new_tag('divisions')
    key = soup.new_tag('key')
    fifths = soup.new_tag('fifths')
    time = soup.new_tag('time')
    beats = soup.new_tag('beats')
    beat_type = soup.new_tag('beat-type')
    staves = soup.new_tag('staves')
    treble_clef = soup.new_tag('clef')
    bass_clef = soup.new_tag('clef')
    treble_sign = soup.new_tag('sign')
    treble_line = soup.new_tag('line')
    bass_sign = soup.new_tag('sign')
    bass_line = soup.new_tag('line')
    attributes.append(divisions)
    divisions.string = '4'
    attributes.append(key)
    key.append(fifths)
    fifths.string = str(key_number)
    attributes.append(time)
    time.append(beats)
    beats.string = str(int(measure_length/4))
    time.append(beat_type)
    beat_type.string = '4'
    attributes.append(staves)
    staves.string = '2'
    attributes.append(treble_clef)
    treble_clef['number']='1'
    treble_clef.append(treble_sign)
    treble_sign.string = 'G'
    treble_clef.append(treble_line)
    treble_line.string = '2'
    attributes.append(bass_clef)
    bass_clef['number'] = '2'
    bass_clef.append(bass_sign)
    bass_sign.string = 'F'
    bass_clef.append(bass_line)
    bass_line.string = '4'
    return attributes


def generate_score(num_measures, measure_length, key_number, rest_prob, chord_probs):
    soup = BeautifulSoup('', 'xml')
    score_partwise = soup.new_tag('score-partwise', version='3.1')
    work = soup.new_tag('work')
    work_title = soup.new_tag('work-title')
    alpha = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890      ')
    text = list(np.random.choice(alpha, size=np.random.randint(8, 25)))
    text = ''.join(text)
    work_title.string = text
    score_partwise.append(work)
    work.append(work_title)
    part_list = soup.new_tag('part-list')
    score_part = soup.new_tag('score-part', id='P1')
    part_name = soup.new_tag('part-name')
    soup.append(score_partwise)
    score_partwise.append(part_list)
    part_list.append(score_part)
    score_part.append(part_name)
    part_name.append('Piano')
    part = soup.new_tag('part', id='P1')
    score_partwise.append(part)

    attributes = generate_attributes(measure_length, key_number)

    for i in range(num_measures):
        measure = generate_measure(measure_length, key_number, rest_prob, chord_probs, i+1)
        if i == 0:
            measure.insert(0, attributes)
        part.append(measure)

    return soup



chord_probs = np.array([5, 1, 5, 5, 5, 5, 1, 100, 1, 5, 5, 5, 5, 1, 5])
chord_probs = chord_probs / chord_probs.sum()
# print(generate_score(16, 4, 3, 0.2, chord_probs).prettify())

# with open('sample_score2.musicxml', 'w+') as f:
#     f.write(str(generate_score(64, 16, 3, 0.2, chord_probs)))