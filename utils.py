import os
import numpy as np
from sequence import EventSeq, ControlSeq
from pretty_midi import *

DEFAULT_SAVING_PROGRAM = instrument_name_to_program('Bright Acoustic Piano')
DEFAULT_DRUM_PROGRAMS = range(27,87)
DEFAULT_LOADING_PROGRAMS = range(128)
DEFAULT_RESOLUTION = 220
DEFAULT_TEMPO = 120
DEFAULT_VELOCITY = 64
DEFAULT_PITCH_RANGE = range(21, 109)
DEFAULT_VELOCITY_RANGE = range(21, 109)
DEFAULT_NORMALIZATION_BASELINE = 60 # C4

# EventSeq ------------------------------------------------------------------------

USE_VELOCITY = True
BEAT_LENGTH = 60 / DEFAULT_TEMPO
DEFAULT_TIME_SHIFT_BINS = 2.8 ** np.arange(32) / 65#1.15 ** np.arange(32) / 65
DEFAULT_VELOCITY_STEPS = 32
DEFAULT_NOTE_LENGTH = BEAT_LENGTH * 2
MIN_NOTE_LENGTH = BEAT_LENGTH / 2


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

def to_midi_file(midi_file_name, drum_notes, classical_notes,program=DEFAULT_SAVING_PROGRAM,
                resolution=DEFAULT_RESOLUTION, tempo=DEFAULT_TEMPO):
    midi = PrettyMIDI(resolution=resolution, initial_tempo=tempo)
    inst0 = Instrument(program, True, 'drum')
    inst0.notes = copy.deepcopy(drum_notes.notes)
    midi.instruments.append(inst0)

    inst1 = Instrument(program, False, 'NoteSeq')
    inst1.notes = copy.deepcopy(classical_notes.notes)
    midi.instruments.append(inst1)
    return midi.write(filename=midi_file_name)

def event_indeces_to_midi_file(event_indeces, midi_file_name, velocity_scale=0.8):
    if len(event_indeces) == 2:
        event_seq = EventSeq.from_array(event_indeces[0])
        note_seq = event_seq.to_note_seq()

        event_seq1 = EventSeq.from_array(event_indeces[1])
        note_seq1 = event_seq1.to_note_seq()
        for note in note_seq.notes:
            note.velocity = int((note.velocity - 64) * velocity_scale + 64)
        for note in note_seq1.notes:
            note.velocity = int((note.velocity - 64) * velocity_scale + 64)
        to_midi_file(midi_file_name, drum_notes=note_seq, classical_notes=note_seq1)

    else:
        event_seq = EventSeq.from_array(event_indeces)
        note_seq = event_seq.to_note_seq()
        for note in note_seq.notes:
            note.velocity = int((note.velocity - 64) * velocity_scale + 64)+24
        note_seq.to_midi_file(midi_file_name,is_drum=False)
    return len(note_seq.notes)

def transposition(events, controls, offset=0):
    # events [steps, batch_size, event_dim]
    # return events, controls

    events = np.array(events, dtype=np.int64)
    controls = np.array(controls, dtype=np.float32)
    event_feat_ranges = EventSeq.feat_ranges()

    on = event_feat_ranges['note_on']
    off = event_feat_ranges['note_off']

    if offset > 0:
        indeces0 = (((on.start <= events) & (events < on.stop - offset)) |
                    ((off.start <= events) & (events < off.stop - offset)))
        indeces1 = (((on.stop - offset  <= events) & (events < on.stop)) |
                    ((off.stop - offset <= events) & (events < off.stop)))
        events[indeces0] += offset
        events[indeces1] += offset - 12
    elif offset < 0:
        indeces0 = (((on.start - offset <= events) & (events < on.stop)) |
                    ((off.start - offset <= events) & (events < off.stop)))
        indeces1 = (((on.start <= events) & (events < on.start - offset)) |
                    ((off.start <= events) & (events < off.start - offset)))
        events[indeces0] += offset
        events[indeces1] += offset + 12

    assert ((0 <= events) & (events < EventSeq.dim())).all()
    histr = ControlSeq.feat_ranges()['pitch_histogram']
    controls[:, :, histr.start:histr.stop] = np.roll(
                    controls[:, :, histr.start:histr.stop], offset, -1)

    return events, controls

def dict2params(d, f=','):
    print(d.items)
    #return f.join(f'{k}={v}' for k, v in d.items())

def params2dict(p, f=',', e='='):
    d = {}
    for item in p.split(f):
        item = item.split(e)
        if len(item) < 2:
            continue
        k, *v = item
        d[k] = eval('='.join(v))
    return d

def compute_gradient_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm
