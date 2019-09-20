from bs4 import BeautifulSoup

def reduce_xml(path):
    with open(path) as f:
        soup = BeautifulSoup(f, 'xml')

    tags_to_keep = soup.find_all(['score-partwise', 'part-list', 'score-part', 'part-name', 'part', 'measure', 'attributes', 'divisions',
                            'key', 'fifths', 'time', 'beats', 'beat-type', 'clef', 'sign', 'line', 'note', 'pitch', 'step', 'alter', 'octave',
                            'duration', 'type', 'rest', 'dot', 'staff', 'notations', 'slur', 'direction', 'direction-type', 'dynamics',
                            'ff', 'f', 'mf', 'mp', 'p', 'pp', 'backup', 'chord'])

    attributes_to_keep = ['version', 'encoding', 'id', 'number', 'type', 'placement']

    for x in soup.find_all():
        if x not in tags_to_keep:
            x.extract()

    for tag in soup.recursiveChildGenerator():
        if 'attrs' in tag.__dict__:
            tag.attrs = {key: tag.attrs[key] for key in tag.attrs if key in attributes_to_keep}

    return soup

reduced = reduce_xml('slur_beam_dynamic_example.musicxml')
with open('slur_beam_dynamic_example_reduced.musicxml', 'w+') as f:
    f.write(str(reduced))
