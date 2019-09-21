from bs4 import BeautifulSoup, Tag, NavigableString


tag_names = ['score-partwise', 'part-list', 'score-part', 'part-name', 'part', 'measure', 'attributes', 'divisions',
             'key', 'fifths', 'time', 'beats', 'beat-type', 'clef', 'sign', 'line', 'note', 'pitch', 'step', 'alter', 'octave',
             'duration', 'type', 'rest', 'dot', 'staff', 'notations', 'slur', 'direction', 'direction-type', 'dynamics',
             'ff', 'f', 'mf', 'mp', 'p', 'pp', 'backup', 'chord']



def fix_pitches(soup):
    pitches = soup.find_all('pitch')
    for pitch in pitches:
        step, alter, octave = pitch.string.split()
        step_tag = Tag(name='step')
        step_tag.string = step
        if alter != '0':
            alter_tag = Tag(name='alter')
            alter_tag.string = alter
        else:
            alter_tag = None
        octave_tag = Tag(name='octave')
        octave_tag.string = octave
        new_pitch = Tag(name='pitch')
        new_pitch.append(step_tag)
        if alter_tag:
            new_pitch.append(alter_tag)
        new_pitch.append(octave_tag)
        pitch.replace_with(new_pitch)
            


def fix_slurs(soup):
    slurs = soup.find_all('slur')
    if len(slurs) % 2 == 1:
        slurs = slurs[:-1]
        slurs[-1].extract()

    i = 0
    while i < len(slurs):
        notations = soup.new_tag('notations')
        new_slur = soup.new_tag('slur')
        new_slur['type'] = 'start'
        notations.append(new_slur)
        slurs[i].replace_with(notations)

        notations = soup.new_tag('notations')
        new_slur = soup.new_tag('slur')
        new_slur['type'] = 'stop'
        notations.append(new_slur)
        slurs[i+1].replace_with(notations)

        i += 2

def fix_dynamics(soup):
    old_dynamics = soup.find_all(['ff', 'f', 'mf', 'mp', 'p', 'pp'])
    for old_dynamic in old_dynamics:
        direction = soup.new_tag('direction')
        direction_type = soup.new_tag('direction-type')
        dynamics = soup.new_tag('dynamics')
        direction.append(direction_type)
        direction_type.append(dynamics)
        dynamics.append(soup.new_tag(old_dynamic.name))
        old_dynamic.replace_with(direction)

def add_measure_numbers(soup):
    measures = soup.find_all('measure')
    for i, measure in enumerate(measures):
        measure['number'] = str(i+1)

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

def restore_attributes(soup, measure_length, key_number):
    first_measure = soup.find('measure')
    attributes = generate_attributes(measure_length, key_number)
    first_measure.insert(0, attributes)
    


def first_split(arr):
        # arr is a list that goes [tag_name, ... '}',...]
        # this returns the two ellipses ...
        counter = 0
        for i, x in enumerate(arr):
            if x in tag_names:
                counter += 1
            if x == '}':
                counter -= 1
            if counter == 0:
                return arr[1:i], arr[i+1:]


def pc_to_xml_helper(pc):
    # returns a list of soup Tags given a pc list
    if not pc:
        return []
    new_tag = Tag(name=pc[0])
    interior, second_part = first_split(pc)
    if interior and interior[0] not in tag_names:
        new_tag.string = ' '.join(interior)
    elif interior and interior[0] in tag_names:
        for child in pc_to_xml_helper(interior):
            new_tag.append(child)
    return [new_tag] + pc_to_xml_helper(second_part)



def pc_to_xml(pc, measure_length, key_number):
    soup = BeautifulSoup(features='xml')
    score_partwise = soup.new_tag('score-partwise', version='3.1')
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
    
    for tag in pc_to_xml_helper(pc):
        part.append(tag)

    fix_slurs(soup)
    fix_dynamics(soup)
    fix_pitches(soup)
    add_measure_numbers(soup)
    restore_attributes(soup, measure_length, key_number)
    return soup




# pc = ['measure', '{', 'note', '{', 'step', '{', 'G', '}', 'octave', '{', '3', '}', '}', 'note', '{', 'step', '{', 'F', '}', 'octave', '{', '4', '}', '}', '}']
# print(pc_to_xml(pc, measure_length=16, key_number=3))


# from xml_to_pc import xml_to_pc
# with open('music_xml/sample_measure.musicxml') as f:
#     soup = BeautifulSoup(f, 'xml')
#     pc = xml_to_pc(soup)
#     xml = pc_to_xml(pc, 12, 3)
    
# with open('music_xml/sample_measure_transformed.musicxml', 'w+') as f:
#     f.write(str(xml))


# pc = ['backup', '{', 'duration', '{', '16', '}', '}']
# print(pc_to_xml_helper(pc))
