import os
import re
import sys
import hashlib
from progress.bar import Bar
import numpy as np

from sequence import NoteSeq, EventSeq, ControlSeq
import utils
from pretty_midi import PrettyMIDI, Note, Instrument
import itertools
import torch


def preprocess_midi(path,programs = range(128)):
    midi_init_file = PrettyMIDI(path)
    notes = itertools.chain(*[
        inst.notes for inst in midi_init_file.instruments
        if inst.program in programs and not inst.is_drum])

    note_seq = NoteSeq(list(notes))
    note_seq.adjust_time(-note_seq.notes[0].start)
    event_seq = EventSeq.from_note_seq(note_seq)
    control_seq = ControlSeq.from_event_seq(event_seq)
    return event_seq.to_array(), control_seq.to_compressed_array()

def preprocess_midi_files_under(midi_root, save_dir):
    midi_paths = list(utils.find_files_by_extensions(midi_root, ['.mid', '.midi']))
    os.makedirs(save_dir, exist_ok=True)
    out_fmt = '{}-{}.data'

    for path in Bar('Processing').iter(midi_paths):
        print(' ', end='[{}]'.format(path), flush=True)

        try:
            data = preprocess_midi(path)
        except KeyboardInterrupt:
            print(' Abort')
            return
        except:
            print(' Error')
            continue

        name = os.path.basename(path)
        code = hashlib.md5(path.encode()).hexdigest()
        save_path = os.path.join(save_dir, out_fmt.format(name, code))
        torch.save(data,save_path)

    print('Done')

if __name__ == '__main__':
    preprocess_midi_files_under(
            midi_root=sys.argv[1],
            save_dir=sys.argv[2])
