import pretty_midi
import os
import numpy as np

def find_files_by_extensions(root, exts=[]):
    def _has_ext(name):
        if not exts:
            return True
        name = name.lower()
        for ext in exts:
            if name.endswith(ext):
                return True
        return False
    for path, _, files in os.walk(root):
        for name in files:
            if _has_ext(name):
                yield os.path.join(path, name)

def parse_track(midi_data):
    track_dict = dict()
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            print(instrument.notes)
        '''if instrument.is_drum:
            track_dict[instrument.name] = instrument.notes
        elif instrument.program in range(0,8):
            track_dict['piano'] = instrument.notes
        '''
for midi_path in list(find_files_by_extensions('./',exts=['ve.mid'])):
    print(midi_path)
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    max_len = 0
    for i in range(len(midi_data.instruments)):
        if not midi_data.instruments[i].is_drum and len(midi_data.instruments[i].notes) > max_len:
            max_len = len(midi_data.instruments[i].notes)
            max_index = i
    '''notes = itertools.chain(*[
        inst.notes for inst in midi.instruments
        if inst.program in programs and not inst.is_drum])'''
    notes = midi_data.instruments[max_index]
    print(max_index)
    drum = pretty_midi.PrettyMIDI()
    drum.instruments.append(notes)
    drum.write('./new_midi.mid')
